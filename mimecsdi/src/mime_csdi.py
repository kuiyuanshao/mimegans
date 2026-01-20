# mime_csdi.py: High-efficiency pipeline for Analog Bits with Self-Conditioning UI.
import torch
import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm

# Ensure script directory is prioritized for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

        # Attributes to be populated during fit()
        self.model = None
        self.norm_stats = None
        self.p1_idx = None
        self.p2_idx = None
        self.weight_idx = None
        self.full_matched_state_vals = None
        self.is_bit = None
        self.cat_p2_groups = None
        self._train_data = None
        self._train_mask = None

    def fit(self, file_path):
        print(f"\n[MIME-CSDI] Initiating Analog Bits Diffusion Pipeline...")

        # Step 1: Preprocessing
        print("   [1/3] Preprocessing: Encoding variables into Analog Bits [-1, 1]...")
        (self._train_data, self._train_mask, self.p1_idx, self.p2_idx,
         self.weight_idx, _, self.norm_stats, self.cat_p2_groups) = process_data(file_path, self.data_info)

        # Step 2: State Matching
        print("   [2/3] Performing State Matching via Residual Variance...")
        self.full_matched_state_vals, self.is_bit = match_state(
            self._train_data, self._train_mask, self.p1_idx, self.p2_idx,
            None, self.weight_idx, self.config,
            n_bootstrap=self.config["else"]["bootstrap"],
            normalization_stats=self.norm_stats,
            data_info=self.data_info
        )

        # Initialize Model
        self.model = CSDI_TwoPhase(
            target_dim=self._train_data.shape[1], config=self.config, device=self.device,
            matched_state=self.full_matched_state_vals, p1_indices=self.p1_idx,
            p2_indices=self.p2_idx, is_bit=self.is_bit, cat_groups=self.cat_p2_groups
        ).to(self.device)

        # Build Residual Pools from Audit Data
        print("   [3/3] Building Empirical Residual Pools and Training...")
        audit_idx = np.where(self._train_mask[:, self.p2_idx[0]] == 1)[0]
        audit_data = self._train_data[audit_idx]

        pool_list = []
        for i in range(len(self.p2_idx)):
            m_step = self.full_matched_state_vals[i]
            sqrt_a_m = np.sqrt(self.model.alpha[m_step])

            # Standardized Residual: (P1 - sqrt(alpha)*P2)
            raw_diff = audit_data[:, self.p1_idx[i]] - sqrt_a_m * audit_data[:, self.p2_idx[i]]
            pool = (raw_diff - np.mean(raw_diff)) / (np.std(raw_diff) + 1e-6)
            pool_list.append(torch.from_numpy(pool).float())

        self.model.residual_pools = torch.stack(pool_list, dim=1).to(self.device)

        # Training Loop
        optimizer = Adam(self.model.parameters(), lr=self.config["train"]["lr"])
        train_loader = DataLoader(
            TwoPhaseDataset(audit_data, self._train_mask[audit_idx]),
            batch_size=self.config["train"]["batch_size"], shuffle=True
        )

        self.model.train()
        for epoch in range(self.config["train"]["epochs"]):
            it = tqdm(train_loader, desc=f"   Epoch {epoch + 1}", mininterval=1, file=sys.stdout)
            for batch in it:
                optimizer.zero_grad()
                loss = self.model(batch["observed_data"].to(self.device))
                loss.backward()
                optimizer.step()
                it.set_postfix(loss=f"{loss.item():.4f}")

        print("[MIME-CSDI] Fit complete.")
        return self

    def impute(self, m=None, save_path="imputed_results.xlsx"):
        m_samples = m if m else self.config["else"]["m"]
        loader = DataLoader(
            TwoPhaseDataset(self._train_data, self._train_mask),
            batch_size=self.config["train"]["batch_size"], shuffle=False
        )

        self.model.eval()
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            for i in range(m_samples):
                print(f"\nGenerating Imputation Sample {i + 1}/{m_samples}...")
                results = []
                with torch.no_grad():
                    for batch in tqdm(loader, desc="   Processing Batches"):
                        sample = self.model.impute_single(
                            batch["observed_data"].to(self.device),
                            batch["observed_mask"].to(self.device)
                        )
                        results.append(sample.cpu())

                full_imputed = torch.cat(results, dim=0).numpy()
                df_imputed = inverse_transform_data(full_imputed, self.norm_stats, self.data_info)
                df_imputed['is_audit'] = (self._train_mask[:, self.p2_idx[0]] == 1).astype(int)
                df_imputed.to_excel(writer, sheet_name=f"Imputation_{i + 1}", index=False)

        print(f"\nSaved {m_samples} samples to {save_path}.")