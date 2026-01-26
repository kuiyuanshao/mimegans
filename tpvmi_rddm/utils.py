# utils.py
import pandas as pd
import numpy as np


def process_data(filepath, data_info):
    """
    Reads data and encodes variables for the Unified Diffusion Model.

    OPTIMIZATION:
    - Applies Log1p transform to ALL numeric variables.
    - Automatically handles negative values by computing a pairwise lower-bound shift.
    - Only adds Phase 2 variables to variable_schema (Phase 1 is implicit condition).
    """
    df = pd.read_csv(filepath)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    p1_vars = data_info['phase1_vars']
    p2_vars = data_info['phase2_vars']
    cat_vars_set = set(data_info.get('cat_vars', []))

    if len(p1_vars) != len(p2_vars):
        raise ValueError("Phase 1 and Phase 2 variable lists must be of equal length.")

    processed_data_list = []
    processed_mask_list = []
    variable_schema = []

    p1_indices = []
    p2_indices = []

    normalization_stats = {}
    current_col_idx = 0

    print(f"\n[Data Processing] Encoding with Log-Transformed Statistics...")

    for p1_name, p2_name in zip(p1_vars, p2_vars):
        if p1_name not in df.columns or p2_name not in df.columns:
            raise ValueError(f"Missing pair: {p1_name} or {p2_name}")

        p1_raw = df[p1_name].values
        p2_raw = df[p2_name].values
        m1 = (~df[p1_name].isna()).values.astype(float)
        m2 = (~df[p2_name].isna()).values.astype(float)

        is_categorical = (p2_name in cat_vars_set) or (p1_name in cat_vars_set)

        if is_categorical:
            # --- CATEGORICAL PROCESSING ---
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

            variable_schema.append({'name': p2_name, 'type': 'categorical', 'num_classes': K})
            normalization_stats[p2_name] = {'type': 'categorical', 'categories': np.array(master_categories)}
            normalization_stats[p1_name] = normalization_stats[p2_name]

        else:
            # --- NUMERIC PROCESSING (Universal Log + Shift) ---
            v1_float = p1_raw.astype(float)
            v2_float = p2_raw.astype(float)

            # 1. Handle Negative Values (Pairwise Lower Bound)
            # Find the global minimum of this specific variable pair to determine shift
            # We use both P1 and P2 to ensure the shift covers the full range of data
            combined_min = np.nanmin(np.concatenate([v1_float, v2_float]))

            shift = 0.0
            if combined_min < 0:
                shift = -combined_min

            # Apply Shift + Log1p
            d1_log = np.log1p(v1_float + shift)
            d2_log = np.log1p(v2_float + shift)

            # 2. Standardize (Z-Score) based on VALID P2 Log Data
            valid_p2_log = d2_log[m2 == 1]

            if len(valid_p2_log) > 0:
                mu, sigma = np.mean(valid_p2_log), np.std(valid_p2_log)
                if sigma < 1e-6: sigma = 1.0
            else:
                mu, sigma = 0.0, 1.0

            d1 = (d1_log - mu) / sigma
            d2 = (d2_log - mu) / sigma

            # Handle NaNs created by log of missing values
            d1 = np.nan_to_num(d1, nan=0.0).reshape(-1, 1)
            d2 = np.nan_to_num(d2, nan=0.0).reshape(-1, 1)

            processed_data_list.extend([d1, d2])
            processed_mask_list.extend([m1.reshape(-1, 1), m2.reshape(-1, 1)])

            variable_schema.append({'name': p2_name, 'type': 'numeric'})

            # Store Shift, Mu, and Sigma for inversion
            normalization_stats[p2_name] = {
                'type': 'numeric',
                'mu': mu,
                'sigma': sigma,
                'shift': shift
            }
            normalization_stats[p1_name] = normalization_stats[p2_name]

        p1_indices.append(current_col_idx)
        p2_indices.append(current_col_idx + 1)
        current_col_idx += 2

    # --- AUXILIARY WEIGHT VARIABLE ---
    weight_idx = None
    w_var = data_info.get('weight_var')
    if w_var:
        if w_var not in df.columns: raise ValueError(f"Weight var {w_var} missing")
        w_raw = df[w_var].values
        m_w = (~df[w_var].isna()).values.astype(float)

        valid_w = w_raw[m_w == 1].astype(float)
        # Weights are usually non-negative, but we treat them as standard numeric here
        # Log transform weights is usually beneficial if they are highly skewed
        w_min = np.min(valid_w) if len(valid_w) > 0 else 0
        w_shift = -w_min if w_min < 0 else 0

        valid_w_log = np.log1p(valid_w + w_shift)

        mu_w = np.mean(valid_w_log) if len(valid_w) > 0 else 0.0
        sig_w = np.std(valid_w_log) if len(valid_w) > 0 else 1.0
        if sig_w < 1e-6: sig_w = 1.0

        w_log = np.log1p(w_raw.astype(float) + w_shift)
        d_w = np.nan_to_num((w_log - mu_w) / sig_w, nan=0.0).reshape(-1, 1)

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
    Reverses Z-Score -> Reverses Log1p -> Reverses Shift.
    """
    reconstructed_df = pd.DataFrame()
    p2_vars = data_info['phase2_vars']

    for i, p2_name in enumerate(p2_vars):
        stats = normalization_stats[p2_name]
        col_data = processed_data[:, i]

        if stats['type'] == 'numeric':
            mu, sigma, shift = stats['mu'], stats['sigma'], stats['shift']

            # 1. Reverse Z-Score
            z_reversed = col_data * sigma + mu

            # 2. Reverse Log1p (Expm1)
            # Clamp to avoid overflow if the model predicts huge values
            z_reversed = np.clip(z_reversed, -50, 50)
            exp_reversed = np.expm1(z_reversed)

            # 3. Reverse Shift
            final_val = exp_reversed - shift

            reconstructed_df[p2_name] = final_val
        else:
            categories = stats['categories']
            indices = np.clip(np.round(col_data), 0, len(categories) - 1).astype(int)
            if len(categories) > 0:
                reconstructed_df[p2_name] = categories[indices]
            else:
                reconstructed_df[p2_name] = indices

    return reconstructed_df