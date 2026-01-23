# tpvmi_rddm.py
import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
import pandas as pd
from torch.optim import Adam
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from networks import RDDM_NET
from utils import inverse_transform_data, process_data


def get_cosine_schedule(num_steps, s=0.008):
    """
    Asymmetric Time Intervals via Cosine Schedule.
    Prevents bits from burning out too early.
    """
    steps = torch.arange(num_steps + 1, dtype=torch.float32) / num_steps
    alpha_bar = torch.cos(((steps + s) / (1 + s)) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    return alpha_bar[1:]  # Return 1 to T


import math


class TPVMI_RDDM:
    def __init__(self, config, data_info, device=None):
        self.config = config
        self.data_info = data_info
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_steps = config["diffusion"]["num_steps"]

        # --- ASYMMETRIC TIME INTERVALS (Cosine Schedule) ---
        # Instead of linear linspace(1, 0), we use Cosine.
        # This keeps the noise level lower for longer, preserving bit structure.
        self.alpha_bars = get_cosine_schedule(self.num_steps).to(self.device)

        # Calculate Betas from Alpha Bars for sampling math
        # beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
        alpha_bar_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)
        self.betas = 1.0 - (self.alpha_bars / alpha_bar_prev)
        self.betas = torch.clamp(self.betas, 0.0001, 0.9999)

        # We generally track beta_bars (cumulative noise) implicitly via alpha_bars in RDDM
        # But for RDDM standard eq, we mostly use alpha_bars.

        self.model = None
        self._global_p1 = None
        self._global_p2 = None
        self._global_aux = None
        self.variable_schema = []

    def fit(self, file_path):
        print(f"\n[TPVMI-RDDM] Training: Analog Bits + ATI (Cosine) + Self-Cond...")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        (proc_data, train_indices, p1_indices, p2_indices, weight_idx, self.variable_schema, self.norm_stats,
         self.raw_df) = process_data(file_path, self.data_info)

        self._global_p1 = []
        self._global_p2 = []
        for (s, e) in p1_indices:
            self._global_p1.append(torch.from_numpy(proc_data[:, s:e]).float().to(self.device))
        for (s, e) in p2_indices:
            self._global_p2.append(torch.from_numpy(proc_data[:, s:e]).float().to(self.device))

        if weight_idx:
            self._global_aux = torch.from_numpy(proc_data[:, weight_idx[0]:weight_idx[1]]).float().to(self.device)
        else:
            self._global_aux = torch.empty((proc_data.shape[0], 0)).float().to(self.device)

        self.model = RDDM_NET(self.config, self.device, self.variable_schema).to(self.device)
        optimizer = Adam(self.model.parameters(), lr=self.config["train"]["lr"])
        batch_size = self.config["train"]["batch_size"]

        num_train = len(train_indices)
        steps_per_epoch = max(1, num_train // batch_size)

        for epoch in range(self.config["train"]["epochs"]):
            self.model.train()
            epoch_losses = []
            it = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}", file=sys.stdout)

            for _ in it:
                b_idx = np.random.choice(train_indices, batch_size)

                b_p1_dict = {v['name']: self._global_p1[i][b_idx] for i, v in enumerate(self.variable_schema) if
                             'aux' not in v['type']}
                b_p2_dict = {v['name']: self._global_p2[i][b_idx] for i, v in enumerate(self.variable_schema) if
                             'aux' not in v['type']}
                b_aux_dict = {}
                for v in self.variable_schema:
                    if 'aux' in v['type']:
                        b_aux_dict[v['name']] = self._global_aux[b_idx]

                optimizer.zero_grad()
                loss = self.calc_unified_loss(b_p1_dict, b_p2_dict, b_aux_dict)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                it.set_postfix(loss=f"{loss.item():.4f}")

        return self

    def calc_unified_loss(self, p1_dict, p2_dict, aux_dict):
        first_var = list(p1_dict.keys())[0]
        B = p1_dict[first_var].shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=self.device).long()

        # Get Schedule for batch
        a_bar = self.alpha_bars[t].view(B, 1)
        # Note: In standard RDDM, 1-a_bar is the noise variance if we assume P1 is center.
        # Analog Bits are [-1, 1]. P1 is [-1, 1].
        # x_t = alpha * P2 + (1-alpha) * P1 + sqrt(1 - alpha^2) * eps ?
        # Or RDDM linear interpolation?
        # RDDM Paper: x_t = alpha * x_0 + (1-alpha) * y + sigma * eps
        # Let's stick to the interpolation logic defined previously but use Cosine alpha.
        # We need a 'b_bar' scaling factor for noise.
        # Typically b_bar approx (1-alpha) or sqrt(1-alpha^2).
        # For simplicity and RDDM consistency:
        # x_t = a_bar * P2 + (1 - a_bar) * P1 + (1 - a_bar) * eps (Simpler noise)
        # OR: x_t = a_bar * P2 + (1 - a_bar) * P1 + sqrt(betas) * eps

        # Let's use a balanced noise scale derived from schedule:
        noise_scale = torch.sqrt(1.0 - a_bar ** 2)  # Standard diffusion noise variance

        x_t_dict = {}
        target_residuals = {}
        target_noise = {}

        # 1. Forward Process
        for var in self.variable_schema:
            if 'aux' in var['type']: continue
            name = var['name']

            p1_val = p1_dict[name]
            p2_val = p2_dict[name]

            eps = torch.randn_like(p2_val)

            # Interpolate + Noise
            x_t = a_bar * p2_val + (1 - a_bar) * p1_val + noise_scale * eps

            x_t_dict[name] = x_t
            target_residuals[name] = (p2_val - p1_val)
            target_noise[name] = eps

        # 2. Self-Conditioning (50% dropout)
        self_cond_dict = None
        if np.random.rand() < 0.5:
            with torch.no_grad():
                out_1 = self.model(x_t_dict, t, p1_dict, aux_dict, self_cond_dict=None)
                self_cond_dict = {}
                for var in self.variable_schema:
                    if 'aux' in var['type']: continue
                    name = var['name']
                    dim = var['dim']
                    pred_res = out_1[name][:, :dim]
                    # Estimate P2_hat = P1 + Res
                    self_cond_dict[name] = (p1_dict[name] + pred_res).detach()

        # 3. Final Prediction
        model_out = self.model(x_t_dict, t, p1_dict, aux_dict, self_cond_dict=self_cond_dict)
        loss_total = 0.0

        for var in self.variable_schema:
            if 'aux' in var['type']: continue
            name = var['name']
            dim = var['dim']

            pred_res = model_out[name][:, :dim]
            pred_eps = model_out[name][:, dim:]

            # Loss: MSE on Residual and Noise
            # Weighting: You can add SNR weighting here, but raw MSE is standard for Bits
            loss_total += F.mse_loss(pred_res, target_residuals[name])
            loss_total += F.mse_loss(pred_eps, target_noise[name])

        return loss_total

    def impute(self, m=None, save_path="imputed_results.xlsx", batch_size=None, eta=0.0):
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        m_s = m if m else self.config["else"]["m"]
        eval_bs = batch_size if batch_size else self.config["train"].get("eval_batch_size", 64)
        N = self._global_p1[0].shape[0]
        self.model.eval()

        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            for samp_i in range(1, m_s + 1):
                print(f"Generating Sample {samp_i}/{m_s}...", flush=True)
                all_p2 = []

                with torch.no_grad():
                    for start in tqdm(range(0, N, eval_bs)):
                        end = min(start + eval_bs, N)
                        B = end - start

                        b_p1_dict = {v['name']: self._global_p1[i][start:end] for i, v in
                                     enumerate(self.variable_schema) if 'aux' not in v['type']}
                        b_aux_dict = {}
                        for v in self.variable_schema:
                            if 'aux' in v['type']:
                                b_aux_dict[v['name']] = self._global_aux[start:end]

                        # Init x_T
                        x_t_dict = {}
                        for name, p1_val in b_p1_dict.items():
                            init_eps = torch.randn_like(p1_val)
                            # Start with low signal (a_T approx 0)
                            start_a = self.alpha_bars[-1]
                            start_noise = torch.sqrt(1.0 - start_a ** 2)
                            x_t_dict[name] = start_a * p1_val + (1 - start_a) * p1_val + start_noise * init_eps

                        # Init Self-Conditioning (Zeros)
                        self_cond_dict = {name: torch.zeros_like(val) for name, val in x_t_dict.items()}

                        # Reverse Loop
                        for t in reversed(range(self.num_steps)):
                            t_b = torch.full((B,), t, device=self.device).long()

                            # Forward with Self-Cond
                            out = self.model(x_t_dict, t_b, b_p1_dict, b_aux_dict, self_cond_dict=self_cond_dict)

                            a_t = self.alpha_bars[t]

                            if t > 0:
                                a_prev = self.alpha_bars[t - 1]
                            else:
                                a_prev = torch.tensor(1.0).to(self.device)

                            # Derive coefficients for update
                            # Standard DDIM/RDDM update step

                            for var in self.variable_schema:
                                if 'aux' in var['type']: continue
                                name = var['name']
                                dim = var['dim']

                                pred_res = out[name][:, :dim]
                                pred_eps = out[name][:, dim:]

                                # 1. Estimate x_0 (P2)
                                p2_hat = b_p1_dict[name] + pred_res

                                # 2. Update Self-Cond for next step
                                self_cond_dict[name] = p2_hat

                                # 3. Compute x_{t-1} using direction to p2_hat
                                # x_{t-1} = a_prev * p2_hat + (1-a_prev) * P1 + noise
                                noise = torch.randn_like(pred_eps)
                                sigma_t = eta * torch.sqrt((1 - a_prev / a_t) * (1 - a_prev) / (1 - a_t))  # DDIM sigma

                                # Simplify: Interpolate towards target
                                dir_p2 = a_prev * p2_hat
                                dir_p1 = (1 - a_prev) * b_p1_dict[name]

                                x_t_dict[name] = dir_p2 + dir_p1 + (0.0 if t == 0 else 0.1 * noise)

                        batch_feats = []
                        for var in self.variable_schema:
                            if 'aux' in var['type']: continue
                            batch_feats.append(x_t_dict[var['name']].cpu())
                        all_p2.append(torch.cat(batch_feats, dim=1))

                full_tensor = torch.cat(all_p2, dim=0).numpy()
                df_p2 = inverse_transform_data(full_tensor, self.norm_stats, self.data_info)

                df_f = self.raw_df.copy()
                for c in df_p2.columns: df_f[c] = df_f[c].fillna(df_p2[c])
                df_f.to_excel(writer, sheet_name=f"Imputation_{samp_i}", index=False)
        print(f"Saved: {save_path}")