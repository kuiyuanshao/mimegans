# mime_csdi.py: Aligned RDDM (SM-Res-N) & Dynamic Absorbing D3PM
# Includes: Masked Loss, Gumbel-Softmax, and Balanced Stratified Sampling

import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
import pandas as pd
from torch.optim import Adam
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from networks import CSDI_Hybrid
from utils import inverse_transform_data, process_data


class MIME_CSDI:
    def __init__(self, config, data_info, device=None):
        self.config = config
        self.data_info = data_info
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_steps = config["diffusion"]["num_steps"]

        # --- Schedules ---
        # 1. Signal Schedule (Alpha Bar)
        self.alpha_bars = torch.linspace(1, 0, self.num_steps).to(self.device)
        # 2. Noise Schedule (Beta Bar)
        beta_start = config["diffusion"].get("beta_start", 0.1)
        beta_end = config["diffusion"].get("beta_end", 0.01)
        self.beta_bars = torch.linspace(beta_start, beta_end, self.num_steps).to(self.device)

        self.model = None

        # Global storage (Only used for Imputation/Validation)
        self._global_p1 = None
        self._global_p2 = None
        self._global_aux = None

        # Schema Indices
        self.num_idxs = None
        self.cat_idxs = None
        self.num_vars = []
        self.cat_vars = []

    def _map_schema_indices(self):
        """Maps schema variable names to column indices once."""
        self.num_vars = [v['name'] for v in self.variable_schema if v['type'] == 'numeric']
        self.cat_vars = [v for v in self.variable_schema if v['type'] == 'categorical']

        num_indices_list, cat_indices_list = [], []
        curr_ptr = 0

        for var in self.variable_schema:
            if 'aux' in var['type']: continue
            if var['type'] == 'numeric':
                num_indices_list.append(curr_ptr)
            elif var['type'] == 'categorical':
                cat_indices_list.append(curr_ptr)
            curr_ptr += 1

        self.num_idxs = torch.tensor(num_indices_list, device=self.device).long()
        self.cat_idxs = torch.tensor(cat_indices_list, device=self.device).long()

    def fit(self, file_path):
        print(f"\n[MIME-CSDI] Training with SM-Res-N (Balanced Stratified Sampling)...")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # 1. Load & Process Data
        (proc_data, proc_mask, p1_idx, p2_idx, weight_idx, self.variable_schema, self.norm_stats,
         self.raw_df) = process_data(file_path, self.data_info)

        # 2. Map Schema
        p1_idx, p2_idx = p1_idx.astype(int), p2_idx.astype(int)
        self._map_schema_indices()

        # 3. Validation Logic
        p2_mask = proc_mask[:, p2_idx]
        valid_rows = np.where(p2_mask.mean(axis=1) > 0.5)[0]

        np.random.shuffle(valid_rows)
        val_ratio = self.config["train"].get("val_ratio", 0.0)
        n_val = int(len(valid_rows) * val_ratio)

        train_rows = valid_rows[n_val:]
        val_rows = valid_rows[:n_val] if n_val > 0 else []

        # 4. Prepare TRAINING Data
        def get_tensors(rows):
            t_p1 = torch.from_numpy(proc_data[rows][:, p1_idx]).float().to(self.device)
            t_p2 = torch.from_numpy(proc_data[rows][:, p2_idx]).float().to(self.device)
            all_idx = set(range(proc_data.shape[1]))
            reserved = set(p1_idx) | set(p2_idx)
            aux_idx = np.array(sorted(list(all_idx - reserved)), dtype=int)
            if len(aux_idx) > 0:
                t_aux = torch.from_numpy(proc_data[rows][:, aux_idx]).float().to(self.device)
            else:
                t_aux = torch.empty((len(rows), 0)).float().to(self.device)
            return t_p1, t_p2, t_aux

        train_p1, train_p2, train_aux = get_tensors(train_rows)

        # --- PREPARE BALANCED SAMPLER INDEX ---
        # We need CPU access to categorical values for fast stratified sampling
        train_cat_map = {}  # {var_name: {class_k: [row_indices]}}

        if len(self.cat_vars) > 0:
            # Extract categorical columns from the training set P2 (Targets)
            # We must use CPU numpy for this part to avoid GPU sync overhead during loop
            p2_cat_cpu = train_p2[:, self.cat_idxs].cpu().numpy().astype(int)

            for k, var in enumerate(self.cat_vars):
                name = var['name']
                num_classes = var['num_classes']
                col_data = p2_cat_cpu[:, k]

                class_indices = {}
                for c in range(num_classes):
                    # Store indices relative to 'train_rows' (0 to N_train)
                    idxs = np.where(col_data == c)[0]
                    if len(idxs) > 0:
                        class_indices[c] = idxs
                train_cat_map[name] = class_indices

        # Store Global Data
        self._global_p1 = torch.from_numpy(proc_data[:, p1_idx]).float().to(self.device)
        self._global_p2 = torch.from_numpy(proc_data[:, p2_idx]).float().to(self.device)
        all_idx = set(range(proc_data.shape[1]))
        reserved = set(p1_idx) | set(p2_idx)
        aux_idx = np.array(sorted(list(all_idx - reserved)), dtype=int)
        if len(aux_idx) > 0:
            self._global_aux = torch.from_numpy(proc_data[:, aux_idx]).float().to(self.device)
        else:
            self._global_aux = torch.empty((proc_data.shape[0], 0)).float().to(self.device)

        del proc_data

        # 5. Model & Opt
        self.model = CSDI_Hybrid(self.config, self.device, self.variable_schema).to(self.device)
        optimizer = Adam(self.model.parameters(), lr=self.config["train"]["lr"])
        batch_size = self.config["train"]["batch_size"]

        # 6. Training Loop
        num_train = len(train_rows)
        steps_per_epoch = max(1, num_train // batch_size)

        for epoch in range(self.config["train"]["epochs"]):
            self.model.train()
            epoch_losses = []

            it = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}", file=sys.stdout)

            for _ in it:
                # --- STRATIFIED SAMPLING LOGIC ---
                if len(train_cat_map) > 0:
                    # 1. Randomly select a categorical variable to condition on
                    target_var_name = np.random.choice(list(train_cat_map.keys()))
                    class_map = train_cat_map[target_var_name]
                    available_classes = list(class_map.keys())

                    # 2. Determine samples per class (Balanced)
                    n_classes = len(available_classes)
                    samples_per_class = batch_size // n_classes
                    remainder = batch_size % n_classes

                    batch_indices = []
                    for i, c in enumerate(available_classes):
                        count = samples_per_class + (1 if i < remainder else 0)
                        # Sample WITH replacement to ensure we can fill the batch even if class is small
                        chosen = np.random.choice(class_map[c], count, replace=True)
                        batch_indices.append(chosen)

                    b_idx = np.concatenate(batch_indices)
                    np.random.shuffle(b_idx)  # Shuffle to mix classes within batch

                else:
                    # Fallback to random sampling if no categorical vars
                    b_idx = np.random.randint(0, num_train, batch_size)

                # Convert to Tensor Index
                b_p1 = train_p1[b_idx]
                b_p2 = train_p2[b_idx]
                b_aux = train_aux[b_idx]

                optimizer.zero_grad()
                loss = self.calc_unified_loss(b_p1, b_p2, b_aux)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                it.set_postfix(loss=f"{loss.item():.4f}")

            if len(val_rows) > 0 and (epoch + 1) % self.config["train"].get("val_interval", 100) == 0:
                v_p1 = self._global_p1[val_rows]
                v_p2 = self._global_p2[val_rows]
                v_aux = self._global_aux[val_rows]
                val_loss = self._validate(v_p1, v_p2, v_aux, batch_size)
                print(f"   [Validation] Epoch {epoch + 1} | Train: {np.mean(epoch_losses):.4f} | Val: {val_loss:.4f}")

        return self

    def _validate(self, p1, p2, aux, bs):
        self.model.eval()
        tl, count = 0, 0
        with torch.no_grad():
            for i in range(0, p1.shape[0], bs):
                end = min(i + bs, p1.shape[0])
                b_p1 = p1[i:end]
                b_p2 = p2[i:end]
                b_aux = aux[i:end]
                loss = self.calc_unified_loss(b_p1, b_p2, b_aux)
                tl += loss.item() * (end - i)
                count += (end - i)
        self.model.train()
        return tl / count

    def calc_unified_loss(self, p1, p2, aux):
        """
        Calculates loss using SM-Res-N Formulation & Masked Categorical Loss.
        """
        B = p1.shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=self.device).long()

        a_bar = self.alpha_bars[t].view(B, 1)
        b_bar = self.beta_bars[t].view(B, 1)

        x_t_dict, p1_dict, aux_dict = {}, {}, {}

        # --- A. Numeric RDDM ---
        if len(self.num_idxs) > 0:
            batch_p2_num = p2[:, self.num_idxs]
            batch_p1_num = p1[:, self.num_idxs]
            true_residual = batch_p2_num - batch_p1_num
            eps = torch.randn_like(batch_p2_num)
            x_t_num = a_bar * batch_p2_num + (1 - a_bar) * batch_p1_num + b_bar * eps
            for i, name in enumerate(self.num_vars):
                x_t_dict[name] = x_t_num[:, i:i + 1]
                p1_dict[name] = batch_p1_num[:, i:i + 1]

        # --- B. Discrete D3PM (Absorbing with Masked Loss) ---
        target_cat_list = []
        if len(self.cat_idxs) > 0:
            batch_p1_cat = p1[:, self.cat_idxs].long()
            batch_p2_cat = p2[:, self.cat_idxs].long()
            probs_matrix = a_bar.expand(B, len(self.cat_vars))
            mask = torch.bernoulli(probs_matrix).bool()
            x_t_indices = torch.where(mask, batch_p2_cat, batch_p1_cat)

            for i, var in enumerate(self.cat_vars):
                name = var['name']
                K = var['num_classes']
                x_t_dict[name] = F.one_hot(x_t_indices[:, i], K).float()
                p1_dict[name] = F.one_hot(batch_p1_cat[:, i], K).float()
                target_cat_list.append(batch_p2_cat[:, i])

        # --- C. Aux ---
        if aux.shape[1] > 0:
            aux_c = 0
            for var in self.variable_schema:
                if 'aux' in var['type']:
                    curr = aux[:, aux_c:aux_c + 1]
                    if var['type'] == 'categorical_aux':
                        aux_dict[var['name']] = F.one_hot(curr.long().squeeze(), var['num_classes']).float()
                    else:
                        aux_dict[var['name']] = curr
                    aux_c += 1

        # --- D. Loss Computation ---
        model_out = self.model(x_t_dict, t, p1_dict, aux_dict)
        loss_total = 0.0

        if len(self.num_vars) > 0:
            pred_stack = torch.stack([model_out[name] for name in self.num_vars], dim=1)
            pred_residual = pred_stack[:, :, 0]
            pred_eps = pred_stack[:, :, 1]
            loss_res = F.mse_loss(pred_residual, true_residual)
            loss_eps = F.mse_loss(pred_eps, eps)
            loss_total += (loss_res + loss_eps)

        if len(self.cat_vars) > 0:
            for i, var in enumerate(self.cat_vars):
                ce_loss = F.cross_entropy(model_out[var['name']], target_cat_list[i], reduction='none')
                is_absorbed = ~mask[:, i]
                masked_loss = ce_loss * is_absorbed.float()
                loss_total += masked_loss.sum() / (is_absorbed.sum() + 1e-6)

        return loss_total

    def impute(self, m=None, save_path="imputed_results.xlsx", batch_size=None, eta=0.0):
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        m_s = m if m else self.config["else"]["m"]
        eval_bs = batch_size if batch_size else self.config["train"].get("eval_batch_size", 64)
        N = self._global_p1.shape[0]
        self.model.eval()

        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            for samp_i in range(1, m_s + 1):
                print(f"Generating Sample {samp_i}/{m_s}...", flush=True)
                all_p2 = []

                with torch.no_grad():
                    for start in tqdm(range(0, N, eval_bs)):
                        end = min(start + eval_bs, N)
                        B = end - start

                        b_p1 = self._global_p1[start:end]
                        b_aux = self._global_aux[start:end]

                        x_t_dict, p1_dict, aux_dict = {}, {}, {}

                        if len(self.num_idxs) > 0:
                            curr_p1_num = b_p1[:, self.num_idxs]
                            init_eps = torch.randn_like(curr_p1_num)
                            start_beta = self.beta_bars[-1]
                            start_alpha = self.alpha_bars[-1]
                            x_T_num = curr_p1_num + start_beta * init_eps

                            for k, name in enumerate(self.num_vars):
                                cp1 = curr_p1_num[:, k:k + 1]
                                x_t_dict[name] = x_T_num[:, k:k + 1]
                                p1_dict[name] = cp1

                        if len(self.cat_idxs) > 0:
                            curr_p1_cat = b_p1[:, self.cat_idxs].long()
                            for k, var in enumerate(self.cat_vars):
                                name = var['name']
                                oh_p1 = F.one_hot(curr_p1_cat[:, k], var['num_classes']).float()
                                x_t_dict[name] = oh_p1.clone()
                                p1_dict[name] = oh_p1

                        if b_aux.shape[1] > 0:
                            aux_c = 0
                            for var in self.variable_schema:
                                if 'aux' in var['type']:
                                    curr = b_aux[:, aux_c:aux_c + 1]
                                    if var['type'] == 'categorical_aux':
                                        aux_dict[var['name']] = F.one_hot(curr.long().squeeze(),
                                                                          var['num_classes']).float()
                                    else:
                                        aux_dict[var['name']] = curr
                                    aux_c += 1

                        for t in reversed(range(self.num_steps)):
                            t_b = torch.full((B,), t, device=self.device).long()
                            out = self.model(x_t_dict, t_b, p1_dict, aux_dict)

                            a_t = self.alpha_bars[t]
                            b_t = self.beta_bars[t]
                            if t > 0:
                                a_prev = self.alpha_bars[t - 1]
                                b_prev = self.beta_bars[t - 1]
                            else:
                                a_prev = torch.tensor(1.0).to(self.device)
                                b_prev = torch.tensor(0.0).to(self.device)

                            sigma_t = eta * b_prev

                            for name in self.num_vars:
                                pred_res = out[name][:, 0:1]
                                pred_eps = out[name][:, 1:2]
                                term_res = (a_t - a_prev) * pred_res
                                var_prev = b_prev ** 2
                                var_sigma = sigma_t ** 2
                                valid_root = torch.sqrt(torch.clamp(var_prev - var_sigma, min=0.0))
                                term_eps = (b_t - valid_root) * pred_eps
                                random_noise = torch.randn_like(pred_eps)
                                term_rand = sigma_t * random_noise
                                x_t_dict[name] = x_t_dict[name] - term_res - term_eps + term_rand

                            for k, var in enumerate(self.cat_vars):
                                name = var['name']
                                logits = out[name]
                                curr_idx = torch.argmax(x_t_dict[name], dim=-1)
                                p1_idx = torch.argmax(p1_dict[name], dim=-1)
                                is_p1 = (curr_idx == p1_idx)
                                chance = (a_prev - a_t) / (1.0 - a_t + 1e-6)
                                chance = torch.clamp(chance, 0.0, 1.0)
                                restore_mask = torch.bernoulli(torch.full((B,), chance, device=self.device)).bool()
                                update_mask = restore_mask & is_p1

                                # Gumbel-Softmax Sampling
                                temp = 0.1 + (t / self.num_steps)
                                new_samples = torch.argmax(F.gumbel_softmax(logits, tau=temp, hard=True), dim=-1)

                                next_indices = torch.where(update_mask, new_samples, curr_idx)

                                # --- BUG FIXED HERE ---
                                if t > 0:
                                    # Use a separate probability for re-masking, NOT 'temp' which can be > 1.0
                                    remask_prob = 0.1 * (t / self.num_steps)
                                    remask_decision = torch.bernoulli(
                                        torch.full((B,), remask_prob, device=self.device)).bool()
                                    next_indices = torch.where(remask_decision, p1_idx, next_indices)

                                x_t_dict[name] = F.one_hot(next_indices, var['num_classes']).float()

                        batch_res = []
                        for var in self.variable_schema:
                            if 'aux' in var['type']: continue
                            final = x_t_dict[var['name']]
                            if var['type'] == 'categorical':
                                final = torch.argmax(final, dim=-1, keepdim=True).float()
                            batch_res.append(final)
                        all_p2.append(torch.cat(batch_res, dim=1).cpu())

                df_p2 = inverse_transform_data(torch.cat(all_p2, dim=0).numpy(), self.norm_stats, self.data_info)
                df_f = self.raw_df.copy()
                for c in df_p2.columns: df_f[c] = df_f[c].fillna(df_p2[c])
                df_f.to_excel(writer, sheet_name=f"Imputation_{samp_i}", index=False)
        print(f"Saved: {save_path}")