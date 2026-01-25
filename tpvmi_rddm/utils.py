# utils.py: Fixed Schema Generation (Only P2 in Schema)
import pandas as pd
import numpy as np


def process_data(filepath, data_info):
    """
    Reads data and encodes variables for the Unified Diffusion Model.

    CRITICAL FIX: Only adds Phase 2 variables to variable_schema.
    Phase 1 variables are implicit conditions that map to the P2 schema entry.
    """
    df = pd.read_csv(filepath)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    p1_vars = data_info['phase1_vars']
    p2_vars = data_info['phase2_vars']

    if len(p1_vars) != len(p2_vars):
        raise ValueError("Phase 1 and Phase 2 variable lists must be of equal length.")

    processed_data_list = []
    processed_mask_list = []
    variable_schema = []

    p1_indices = []
    p2_indices = []

    normalization_stats = {}
    current_col_idx = 0

    print(f"\n[Data Processing] Encoding with Paired Statistics...")

    for p1_name, p2_name in zip(p1_vars, p2_vars):
        if p1_name not in df.columns or p2_name not in df.columns:
            raise ValueError(f"Missing pair: {p1_name} or {p2_name}")

        p1_raw = df[p1_name].values
        p2_raw = df[p2_name].values
        m1 = (~df[p1_name].isna()).values.astype(float)
        m2 = (~df[p2_name].isna()).values.astype(float)

        is_categorical = (p2_name in data_info.get('cat_vars', [])) or (p1_name in data_info.get('cat_vars', []))

        # --- DATA PROCESSING (Same as before) ---
        if is_categorical:
            u1 = pd.unique(p1_raw[pd.notna(p1_raw)])
            u2 = pd.unique(p2_raw[pd.notna(p2_raw)])
            unique_set = set(u1) | set(u2)
            master_categories = sorted(list(unique_set))

            cat_to_int = {val: i for i, val in enumerate(master_categories)}
            K = len(master_categories)
            if K == 0: K = 1

            d1 = np.array([cat_to_int.get(x, 0) for x in p1_raw]).reshape(-1, 1)
            d2 = np.array([cat_to_int.get(x, 0) for x in p2_raw]).reshape(-1, 1)

            processed_data_list.extend([d1, d2])
            processed_mask_list.extend([m1.reshape(-1, 1), m2.reshape(-1, 1)])

            # --- SCHEMA FIX: ONLY APPEND P2 ---
            variable_schema.append({'name': p2_name, 'type': 'categorical', 'num_classes': K})
            normalization_stats[p2_name] = {'type': 'categorical', 'categories': np.array(master_categories)}
            # Map P1 to P2 stats for consistency if needed manually later
            normalization_stats[p1_name] = normalization_stats[p2_name]

        else:
            valid_p2 = p2_raw[m2 == 1].astype(float)
            if len(valid_p2) > 0:
                mu, sigma = np.mean(valid_p2), np.std(valid_p2)
                if sigma < 1e-6: sigma = 1.0
            else:
                mu, sigma = 0.0, 1.0

            d1 = ((p1_raw.astype(float) - mu) / sigma)
            d2 = ((p2_raw.astype(float) - mu) / sigma)
            d1 = np.nan_to_num(d1, nan=0.0).reshape(-1, 1)
            d2 = np.nan_to_num(d2, nan=0.0).reshape(-1, 1)

            processed_data_list.extend([d1, d2])
            processed_mask_list.extend([m1.reshape(-1, 1), m2.reshape(-1, 1)])

            # --- SCHEMA FIX: ONLY APPEND P2 ---
            variable_schema.append({'name': p2_name, 'type': 'numeric'})

            normalization_stats[p2_name] = {'type': 'numeric', 'mu': mu, 'sigma': sigma}
            normalization_stats[p1_name] = normalization_stats[p2_name]

        p1_indices.append(current_col_idx)
        p2_indices.append(current_col_idx + 1)
        current_col_idx += 2

    # Aux Weight Var
    weight_idx = None
    w_var = data_info.get('weight_var')
    if w_var:
        if w_var not in df.columns: raise ValueError(f"Weight var {w_var} missing")
        w_raw = df[w_var].values
        m_w = (~df[w_var].isna()).values.astype(float)

        valid_w = w_raw[m_w == 1].astype(float)
        mu_w = np.mean(valid_w) if len(valid_w) > 0 else 0.0
        sig_w = np.std(valid_w) if len(valid_w) > 0 else 1.0
        if sig_w < 1e-6: sig_w = 1.0

        d_w = np.nan_to_num((w_raw.astype(float) - mu_w) / sig_w, nan=0.0).reshape(-1, 1)

        processed_data_list.append(d_w)
        processed_mask_list.append(m_w.reshape(-1, 1))

        variable_schema.append({'name': w_var, 'type': 'numeric_aux'})
        weight_idx = current_col_idx
        current_col_idx += 1

    final_data = np.hstack(processed_data_list)
    final_mask = np.hstack(processed_mask_list)

    return (final_data, final_mask, np.array(p1_indices), np.array(p2_indices),
            weight_idx, variable_schema, normalization_stats, df)


def inverse_transform_data(processed_data, normalization_stats, data_info):
    """
    Decodes ONLY Phase 2 variables from the model output.
    """
    reconstructed_df = pd.DataFrame()
    p2_vars = data_info['phase2_vars']

    for i, p2_name in enumerate(p2_vars):
        stats = normalization_stats[p2_name]
        col_data = processed_data[:, i]

        if stats['type'] == 'numeric':
            mu, sigma = stats['mu'], stats['sigma']
            reconstructed_df[p2_name] = col_data * sigma + mu
        else:
            categories = stats['categories']
            indices = np.clip(np.round(col_data), 0, len(categories) - 1).astype(int)
            if len(categories) > 0:
                reconstructed_df[p2_name] = categories[indices]
            else:
                reconstructed_df[p2_name] = indices

    return reconstructed_df