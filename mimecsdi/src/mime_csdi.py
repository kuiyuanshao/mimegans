# mime_csdi.py: Pipeline with optimized Multiple Imputation loop and UI reporting.
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
from networks import CSDI_TwoPhase
from utils import inverse_transform_data, process_data
from match_state import match_state


class TwoPhaseDataset(Dataset):
    def __init__(self, data, mask):
        self.data = torch.FloatTensor(data)
        self.mask = torch.FloatTensor(mask)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return {"observed_data": self.data[idx], "observed_mask": self.mask[idx]}


class MIME_CSDI:
    def __init__(self, config, data_info, device=None):
        self.config = config
        self.data_info = data_info
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.norm_stats = None, None
        self.p1_idx, self.p2_idx, self.weight_idx = None, None, None
        self.full_matched_state_vals, self.is_binary = None, None
        self.cat_p2_groups, self._train_data, self._train_mask = None, None, None

    def fit(self, file_path):
        print(f"\n[MIME-CSDI] Fit initiated...")
        self._step_preprocess(file_path)
        self._step_match()
        self._step_train()
        print("[MIME-CSDI] Fit complete.")
        return self

    def _step_preprocess(self, file_path):
        (self._train_data, self._train_mask, self.p1_idx, self.p2_idx,
         self.weight_idx, _, self.norm_stats, self.cat_p2_groups) = process_data(file_path, self.data_info)

    def _step_match(self):
        print("   [2/3] State Matching...")
        res = match_state(self._train_data, self._train_mask, self.p1_idx, self.p2_idx,
                          None, self.weight_idx, self.config,
                          n_bootstrap=self.config["else"]["bootstrap"],
                          normalization_stats=self.norm_stats)
        self.full_matched_state_vals, self.is_binary = res

    def _step_train(self):
        print("   [3/3] Training Diffusion Model...")
        self._build_model_arch()
        audit_idx = np.where(self._train_mask[:, self.p2_idx[0]] == 1)[0]
        audit_data = self._train_data[audit_idx]
        pool_list = []
        for i, p2_col in enumerate(self.p2_idx):
            p1_col, m_step = self.p1_idx[i], self.full_matched_state_vals[i]
            sqrt_a_m = np.sqrt(self.model.alpha[m_step])  # FIX: self.model.alpha is numpy
            raw_diff = audit_data[:, p1_col] - sqrt_a_m * audit_data[:, p2_col]
            if self.is_binary[i]:
                pool = ((audit_data[:, p1_col] > 0.5) != (audit_data[:, p2_col] > 0.5)).astype(np.float32)
            else:
                pool = ((raw_diff - np.mean(raw_diff)) / (np.std(raw_diff) + 1e-6)).astype(np.float32)
            pool_list.append(torch.from_numpy(pool))
        self.model.residual_pools = torch.stack(pool_list, dim=1).to(self.device)
        train_loader = DataLoader(TwoPhaseDataset(audit_data, self._train_mask[audit_idx]),
                                  batch_size=self.config["train"]["batch_size"], shuffle=True)
        optimizer = Adam(self.model.parameters(), lr=self.config["train"]["lr"])
        self.model.train()
        for epoch in range(self.config["train"]["epochs"]):
            with tqdm(train_loader, desc=f"   Epoch {epoch + 1}", mininterval=1, file=sys.stdout) as it:
                for batch in it:
                    optimizer.zero_grad()
                    loss = self.model(batch["observed_data"].to(self.device))
                    loss.backward()
                    optimizer.step()

    def _build_model_arch(self):
        self.model = CSDI_TwoPhase(target_dim=self._train_data.shape[1], config=self.config, device=self.device,
                                   matched_state=self.full_matched_state_vals, p1_indices=self.p1_idx,
                                   p2_indices=self.p2_idx, is_binary=self.is_binary,
                                   cat_groups=self.cat_p2_groups).to(self.device)

    def impute(self, m=None, save_path="imputed_results.xlsx"):
        m_samples = m if m else self.config["else"]["m"]
        loader = DataLoader(TwoPhaseDataset(self._train_data, self._train_mask),
                            batch_size=self.config["train"]["batch_size"], shuffle=False)
        self.model.eval()

        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            for i in range(m_samples):
                print(f"\nMultiple Imputation Sample {i + 1}/{m_samples}:")
                results_for_this_m = []
                with torch.no_grad():
                    for batch in tqdm(loader, desc="   Processing Batches", leave=True, dynamic_ncols=True,
                                      file=sys.stdout):
                        sample = self.model.impute_single(batch["observed_data"].to(self.device),
                                                          batch["observed_mask"].to(self.device))
                        results_for_this_m.append(sample.cpu())

                full_imputed_norm = torch.cat(results_for_this_m, dim=0).numpy()
                df_imputed = inverse_transform_data(full_imputed_norm, self.norm_stats, self.data_info)
                df_imputed['is_audit'] = (self._train_mask[:, self.p2_idx[0]] == 1).astype(int)
                df_imputed.to_excel(writer, sheet_name=f"Imputation_{i + 1}", index=False)

        print(f"\nSuccessfully saved {m_samples} sheets to {save_path}.")
        return None