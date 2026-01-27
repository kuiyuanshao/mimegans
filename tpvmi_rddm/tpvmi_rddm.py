import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
import numpy as np
import pandas as pd
import copy
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from networks import RDDM_NET
from utils import inverse_transform_data, process_data


# ==========================================
# 1. Full SWAG Implementation (Low-Rank + Diag)
# ==========================================
class FullSWAG(nn.Module):
    def __init__(self, base_model, max_num_models=20, var_clamp=1e-30):
        super(FullSWAG, self).__init__()
        self.base = copy.deepcopy(base_model)
        self.base.train()
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp
        self.n_models = torch.zeros([1], dtype=torch.long)
        self.params = list()

        for name, param in self.base.named_parameters():
            safe_name = name.replace(".", "_")
            self.register_buffer(f"{safe_name}_mean", torch.zeros_like(param.data))
            self.register_buffer(f"{safe_name}_sq_mean", torch.zeros_like(param.data))
            self.register_buffer(f"{safe_name}_cov_mat_sqrt", torch.empty(0, param.numel()))
            self.params.append((name, safe_name, param))

    def collect_model(self, base_model):
        curr_params = dict(base_model.named_parameters())
        n = self.n_models.item()
        for name, safe_name, _ in self.params:
            if name not in curr_params: continue
            param = curr_params[name]
            mean = getattr(self, f"{safe_name}_mean")
            sq_mean = getattr(self, f"{safe_name}_sq_mean")
            cov_mat_sqrt = getattr(self, f"{safe_name}_cov_mat_sqrt")

            mean = mean * n / (n + 1.0) + param.data.to(mean.device) / (n + 1.0)
            sq_mean = sq_mean * n / (n + 1.0) + (param.data.to(sq_mean.device) ** 2) / (n + 1.0)
            dev = (param.data.to(mean.device) - mean).view(-1, 1)
            cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.t()), dim=0)

            if (cov_mat_sqrt.size(0)) > self.max_num_models:
                cov_mat_sqrt = cov_mat_sqrt[1:, :]

            setattr(self, f"{safe_name}_mean", mean)
            setattr(self, f"{safe_name}_sq_mean", sq_mean)
            setattr(self, f"{safe_name}_cov_mat_sqrt", cov_mat_sqrt)
        self.n_models.add_(1)

    def sample(self, scale=1, cov=True):
        scale_sqrt = scale ** 0.5
        for name, safe_name, base_param in self.params:
            mean = getattr(self, f"{safe_name}_mean")
            sq_mean = getattr(self, f"{safe_name}_sq_mean")
            cov_mat_sqrt = getattr(self, f"{safe_name}_cov_mat_sqrt")

            var = torch.clamp(sq_mean - mean ** 2, min=self.var_clamp)
            var_sample = var.sqrt() * torch.randn_like(var)

            if cov and cov_mat_sqrt.size(0) > 0:
                K = cov_mat_sqrt.size(0)
                z2 = torch.randn(K, 1, device=mean.device)
                cov_sample = cov_mat_sqrt.t().matmul(z2).view_as(mean)
                cov_sample /= (self.max_num_models - 1) ** 0.5
                rand_sample = var_sample + cov_sample
            else:
                rand_sample = var_sample

            sample = mean + scale_sqrt * rand_sample
            base_param.data.copy_(sample)
        return self.base


class TPVMI_RDDM:
    def __init__(self, config, data_info, device=None):
        self.config = config
        self.data_info = data_info
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_steps = config["diffusion"]["num_steps"]

        # Schedules
        if config["diffusion"]["schedule"] == "linear":
            self.alpha_bars = torch.linspace(1, 0, self.num_steps).to(self.device)
        elif config["diffusion"]["schedule"] == "cosine":
            self.alpha_bars = self._get_cosine_schedule(self.num_steps).to(self.device)

        # [FIX]: Increase beta_end default from 0.01 to 0.1
        # This increases the starting noise at t=T, giving the model more freedom
        # to move away from P1 (Proxy) and avoid "Ghost traps".
        beta_start = config["diffusion"].get("beta_start", 0.0001)
        beta_end = config["diffusion"].get("beta_end", 0.1)
        self.beta_bars = torch.linspace(beta_start, beta_end, self.num_steps).to(self.device)

        self.model = None
        self.model_list = []
        self.swag_model = None
        self._global_p1 = None
        self._global_p2 = None
        self._global_aux = None
        self.num_idxs = None
        self.cat_idxs = None
        self.num_vars = []
        self.cat_vars = []

    def _get_cosine_schedule(self, num_steps, s=0.008):
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        return alphas_cumprod[1:]

    def _map_schema_indices(self):
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
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        lr = self.config["train"]["lr"]
        epochs = self.config["train"]["epochs"]
        batch_size = self.config["train"]["batch_size"]
        mi_approx = self.config["else"]["mi_approx"]

        (proc_data, proc_mask, p1_idx, p2_idx, weight_idx, self.variable_schema, self.norm_stats,
         self.raw_df) = process_data(file_path, self.data_info)

        p1_idx, p2_idx = p1_idx.astype(int), p2_idx.astype(int)
        self._map_schema_indices()

        p2_mask = proc_mask[:, p2_idx]
        valid_rows = np.where(p2_mask.mean(axis=1) > 0.5)[0]

        self._global_p1 = torch.from_numpy(proc_data[:, p1_idx]).float().to(self.device)
        self._global_p2 = torch.from_numpy(proc_data[:, p2_idx]).float().to(self.device)
        all_idx = set(range(proc_data.shape[1]))
        reserved = set(p1_idx) | set(p2_idx)
        aux_idx = np.array(sorted(list(all_idx - reserved)), dtype=int)
        if len(aux_idx) > 0:
            self._global_aux = torch.from_numpy(proc_data[:, aux_idx]).float().to(self.device)
        else:
            self._global_aux = torch.empty((proc_data.shape[0], 0)).float().to(self.device)

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

        if self.config["else"]["mi_approx"] == "bootstrap":
            num_train_mods = self.config["else"]["m"]
        else:
            num_train_mods = 1

        for k in range(num_train_mods):
            print(f"\n[TPVMI-RDDM] Training {k + 1}/{num_train_mods}...")
            rng = np.random.default_rng()
            if self.config["else"]["mi_approx"] == "bootstrap":
                current_rows = rng.choice(valid_rows, size=len(valid_rows), replace=True)
            else:
                current_rows = np.random.permutation(valid_rows)
            val_ratio = self.config["train"].get("val_ratio", 0.0)
            n_val = int(len(current_rows) * val_ratio)
            train_rows, val_rows = current_rows[n_val:], current_rows[:n_val] if n_val > 0 else []

            train_p1, train_p2, train_aux = get_tensors(train_rows)
            self.model = RDDM_NET(self.config, self.device, self.variable_schema).to(self.device)

            if mi_approx == "SWAG":
                swa_start = int(epochs * 0.80)
                optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9)
                scheduler = CosineAnnealingLR(optimizer, T_max=swa_start, eta_min=lr * 0.5)
                self.swag_model = FullSWAG(self.model, max_num_models=int(self.config["else"]["m"] * 5)).to(self.device)
            else:
                optimizer = Adam(self.model.parameters(), lr=self.config["train"]["lr"])

            num_train = len(train_rows)
            steps_per_epoch = math.ceil(num_train / batch_size)

            for epoch in range(epochs):
                self.model.train()
                epoch_losses = []

                # Shuffle indices once per epoch for non-bootstrap modes
                if mi_approx != "bootstrap":
                    epoch_indices = np.random.permutation(num_train)

                it = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}", file=sys.stdout)

                if mi_approx == "SWAG":
                    is_swa_phase = (epoch >= swa_start)
                    if is_swa_phase:
                        swa_lr = lr * 0.5
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = swa_lr

                for step in it:
                    if mi_approx == "bootstrap":
                        b_idx = np.random.randint(0, num_train, batch_size)
                    else:
                        start_idx = step * batch_size
                        end_idx = min(start_idx + batch_size, num_train)
                        if start_idx >= num_train: break
                        b_idx = epoch_indices[start_idx:end_idx]

                    b_p1 = train_p1[b_idx]
                    b_p2 = train_p2[b_idx]
                    b_aux = train_aux[b_idx]

                    optimizer.zero_grad()
                    loss = self.calc_unified_loss(b_p1, b_p2, b_aux)
                    loss.backward()

                    if mi_approx == "SWAG":
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

                    optimizer.step()
                    epoch_losses.append(loss.item())
                    it.set_postfix(loss=f"{loss.item():.4f}")

                if mi_approx == "SWAG":
                    if not is_swa_phase: scheduler.step()
                    if is_swa_phase: self.swag_model.collect_model(self.model)

                if len(val_rows) > 0 and (epoch + 1) % self.config["train"].get("val_interval", 100) == 0:
                    v_p1 = self._global_p1[val_rows]
                    v_p2 = self._global_p2[val_rows]
                    v_aux = self._global_aux[val_rows]
                    val_loss = self._validate(v_p1, v_p2, v_aux, batch_size)
                    phase_str = "SWAG" if is_swa_phase else "Pre-train"
                    print(f"   [Validation] Epoch {epoch + 1} | Val: {val_loss:.4f} | Phase: {phase_str}")
            self.model_list.append(self.model)
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
        B = p1.shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=self.device).long()
        a_bar = self.alpha_bars[t].view(B, 1)
        b_bar = self.beta_bars[t].view(B, 1)

        x_t_dict, p1_dict, aux_dict, mask_dict = {}, {}, {}, {}

        # --- NUMERIC ---
        if len(self.num_idxs) > 0:
            batch_p2_num = p2[:, self.num_idxs]
            batch_p1_num = p1[:, self.num_idxs]
            true_residual = batch_p2_num - batch_p1_num
            eps = torch.randn_like(batch_p2_num)

            # centered_res = true_residual - true_residual.mean(dim=0, keepdim=True)
            # cov_matrix = torch.matmul(centered_res.T, centered_res) / (B - 1 + 1e-6)
            # cov_matrix = cov_matrix + torch.eye(cov_matrix.shape[0], device=self.device) * 1e-4
            # L = torch.diag(torch.sqrt(torch.diagonal(cov_matrix)))
            #
            # structured_eps = torch.matmul(eps, L.t())
            # idx_shuf = torch.randperm(B, device=self.device)
            # shuffled_residual = true_residual[idx_shuf]
            # shuffle_prob = 0.2
            # mask = (torch.rand((B, 1), device=self.device) > shuffle_prob).float()
            # hybrid_residual = mask * true_residual + (1 - mask) * shuffled_residual
            # x_t_num = batch_p1_num + a_bar * hybrid_residual + b_bar * eps

            x_t_num = a_bar * batch_p2_num + (1 - a_bar) * batch_p1_num + b_bar * eps
            for i, name in enumerate(self.num_vars):
                x_t_dict[name] = x_t_num[:, i:i + 1]
                p1_dict[name] = batch_p1_num[:, i:i + 1]
                mask_dict[name] = torch.ones((B, 1), device=self.device)

        # --- CATEGORICAL ---
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
                mask_dict[name] = mask[:, i:i + 1].float()

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

        model_out = self.model(x_t_dict, t, p1_dict, aux_dict, mask_dict)
        loss_total = 0.0

        if len(self.num_vars) > 0:
            pred_stack = torch.stack([model_out[name] for name in self.num_vars], dim=1)

            # pred_res = pred_stack[:, :, 0]
            # pred_eps = pred_stack[:, :, 1]
            # pred_gate_logits = pred_stack[:, :, 2]
            #
            # threshold = 0.05
            # is_shift_target = (torch.abs(true_residual) > threshold).float()
            # ghost_mask = (is_shift_target > 0.5).squeeze()

            # if ghost_mask.sum() > 0:
            #     res_ghosts = pred_res[ghost_mask]
            #     true_res_ghosts = true_residual[ghost_mask]
            #     res_loss = F.mse_loss(res_ghosts, true_res_ghosts)
            # else:
            #     res_loss = torch.tensor(0.0, device=self.device)

            # gate_loss = F.binary_cross_entropy_with_logits(pred_gate_logits, is_shift_target)
            # eps_loss = F.mse_loss(pred_eps, eps)

            # Weighting: Prioritize accurate Gating and Residuals over Noise
            # loss_total += (10.0 * res_loss + 5.0 * gate_loss + eps_loss)

            loss_total += (F.mse_loss(pred_stack[:, :, 0], true_residual) +
                           F.mse_loss(pred_stack[:, :, 1], eps))

        if len(self.cat_vars) > 0:
            for i, var in enumerate(self.cat_vars):
                ce_loss = F.cross_entropy(model_out[var['name']], target_cat_list[i], reduction='none')
                is_absorbed = ~mask[:, i]
                masked_loss = ce_loss * is_absorbed.float()
                loss_total += (masked_loss.sum() / (is_absorbed.sum() + 1e-6))

        return loss_total

    def impute(self, m=None, save_path="imputed_results.parquet", batch_size=None, eta=0.0):
        if not save_path.endswith('.parquet'):
            base = os.path.splitext(save_path)[0]
            save_path = f"{base}.parquet"
            print(f"Note: File extension changed to .parquet: {save_path}")

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        pd.set_option('future.no_silent_downcasting', True)

        m_s = m if m else self.config["else"]["m"]
        eval_bs = batch_size if batch_size else self.config["train"].get("eval_batch_size", 64)
        N = self._global_p1.shape[0]

        if self.config["else"]["mi_approx"] == "SWAG":
            if self.swag_model is None or self.swag_model.n_models.item() == 0:
                print("Warning: No SWAG models collected. Using base model point estimate.")
            else:
                print(f"Using Full SWAG for sampling (Collected {self.swag_model.n_models.item()} models)")

        all_imputed_dfs = []

        for samp_i in range(1, m_s + 1):
            print(f"\nGenerating Sample {samp_i}/{m_s}...", flush=True)

            if self.config["else"]["mi_approx"] == "SWAG":
                if self.swag_model is not None and self.swag_model.n_models.item() > 1:
                    model_to_use = self.swag_model.sample(scale=1, cov=True)
                    model_to_use.eval()
            elif self.config["else"]["mi_approx"] == "dropout":
                model_to_use = self.model_list[0]
                model_to_use.eval()
                for modu in model_to_use.modules():
                    if modu.__class__.__name__.startswith('Dropout'):
                        modu.train()
            elif self.config["else"]["mi_approx"] == "bootstrap":
                model_to_use = self.model_list[samp_i - 1]
                model_to_use.eval()
            else:
                model_to_use = self.model_list[0]
                model_to_use.eval()
            all_p2 = []

            with torch.no_grad():
                for start in tqdm(range(0, N, eval_bs)):
                    end = min(start + eval_bs, N)
                    B = end - start
                    b_p1 = self._global_p1[start:end]
                    b_aux = self._global_aux[start:end]

                    x_t_dict, p1_dict, aux_dict, mask_dict = {}, {}, {}, {}
                    if len(self.num_idxs) > 0:
                        curr_p1_num = b_p1[:, self.num_idxs]
                        init_eps = torch.randn_like(curr_p1_num)
                        start_beta = self.beta_bars[-1]
                        x_T_num = curr_p1_num + start_beta * init_eps
                        for k, name in enumerate(self.num_vars):
                            cp1 = curr_p1_num[:, k:k + 1]
                            x_t_dict[name] = x_T_num[:, k:k + 1]
                            p1_dict[name] = cp1
                            mask_dict[name] = torch.ones((B, 1), device=self.device)

                    if len(self.cat_idxs) > 0:
                        curr_p1_cat = b_p1[:, self.cat_idxs].long()
                        for k, var in enumerate(self.cat_vars):
                            name = var['name']
                            oh_p1 = F.one_hot(curr_p1_cat[:, k], var['num_classes']).float()
                            x_t_dict[name] = oh_p1.clone()
                            p1_dict[name] = oh_p1
                            mask_dict[name] = torch.zeros((B, 1), device=self.device)

                    if b_aux.shape[1] > 0:
                        aux_c = 0
                        for var in self.variable_schema:
                            if 'aux' in var['type']:
                                curr = b_aux[:, aux_c:aux_c + 1]
                                if var['type'] == 'categorical_aux':
                                    aux_dict[var['name']] = F.one_hot(curr.long().squeeze(), var['num_classes']).float()
                                else:
                                    aux_dict[var['name']] = curr
                                aux_c += 1

                    for t in reversed(range(self.num_steps)):
                        t_b = torch.full((B,), t, device=self.device).long()
                        out = model_to_use(x_t_dict, t_b, p1_dict, aux_dict, mask_dict)

                        a_t = self.alpha_bars[t]
                        b_t = self.beta_bars[t]
                        a_prev = self.alpha_bars[t - 1] if t > 0 else torch.tensor(1.0).to(self.device)
                        b_prev = self.beta_bars[t - 1] if t > 0 else torch.tensor(0.0).to(self.device)
                        sigma_t = eta * b_prev

                        # Numeric Update
                        for name in self.num_vars:
                            pred_res = out[name][:, 0:1]
                            pred_eps = out[name][:, 1:2]

                            term_res = (a_t - a_prev) * pred_res
                            valid_root = torch.sqrt(torch.clamp(b_prev ** 2 - sigma_t ** 2, min=0.0))
                            term_eps = (b_t - valid_root) * pred_eps

                            noise = torch.randn_like(pred_eps)
                            x_t_dict[name] = x_t_dict[name] - term_res - term_eps + sigma_t * noise

                        # Categorical Update
                        for k, var in enumerate(self.cat_vars):
                            name = var['name']
                            K = var['num_classes']
                            logits = out[name]
                            pred_x0_probs = F.softmax(logits, dim=-1)
                            curr_x_t = x_t_dict[name]
                            if curr_x_t.dim() == 1: curr_x_t = F.one_hot(curr_x_t.long(), K).float()

                            prob_reveal = (a_prev - a_t) / (1.0 - a_t + 1e-6)
                            prob_reveal = torch.clamp(prob_reveal, 0.0, 1.0)
                            if prob_reveal.dim() == 0:
                                prob_reveal = prob_reveal.view(1, 1)
                            elif prob_reveal.dim() == 1:
                                prob_reveal = prob_reveal.unsqueeze(-1)

                            posterior_probs = (1.0 - prob_reveal) * curr_x_t + prob_reveal * pred_x0_probs
                            posterior_probs = posterior_probs / (posterior_probs.sum(dim=-1, keepdim=True) + 1e-8)
                            x_next_indices = torch.multinomial(posterior_probs, 1).squeeze(-1)
                            x_t_dict[name] = F.one_hot(x_next_indices, K).float()

                    batch_res = []
                    for var in self.variable_schema:
                        if 'aux' in var['type']: continue
                        if var['type'] == 'categorical':
                            final = torch.argmax(x_t_dict[var['name']], dim=-1, keepdim=True).float()
                            batch_res.append(final)
                        else:
                            batch_res.append(x_t_dict[var['name']])
                    all_p2.append(torch.cat(batch_res, dim=1).cpu())

            df_p2 = inverse_transform_data(torch.cat(all_p2, dim=0).numpy(), self.norm_stats, self.data_info)
            df_f = self.raw_df.copy()

            for c in df_p2.columns:
                if c in df_f.columns:
                    df_f[c] = df_f[c].fillna(df_p2[c])

            df_f.insert(0, "imp_id", samp_i)
            all_imputed_dfs.append(df_f)

        final_df = pd.concat(all_imputed_dfs, ignore_index=True)
        final_df['imp_id'] = final_df['imp_id'].astype(int)
        final_df.to_parquet(save_path, index=False)
        print(f"Saved stacked imputations to: {save_path} (Shape: {final_df.shape})")