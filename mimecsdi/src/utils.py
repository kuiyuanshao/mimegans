# utils.py: Data processing utilities for reading, cleaning, encoding, and normalizing two-phase data.
import pandas as pd
import numpy as np


def process_data(filepath, data_info):
    """
    Reads data, cleans, encodes, and normalizes.
    Numeric features are Z-scored; Categorical features are kept as 0/1 binaries.
    """
    df = pd.read_csv(filepath)

    p1_vars = data_info['phase1_vars']
    p2_vars = data_info['phase2_vars']
    cat_vars_set = set(data_info.get('cat_vars', []))
    weight_var = data_info.get('weight_var')

    processed_data_list = []
    processed_mask_list = []
    p1_indices = []
    p2_indices = []
    p1_generated_names = []
    cat_p2_groups = []

    normalization_stats = {}
    current_col_idx = 0

    def get_unified_string_series(s):
        s_num = pd.to_numeric(s, errors='coerce')
        if s_num.notna().any():
            non_na = s_num.dropna()
            if (non_na % 1 == 0).all():
                return s_num.astype('Int64').astype(str).replace('<NA>', np.nan)
        return s.astype(str).str.strip().replace('nan', np.nan)

    for p1_name, p2_name in zip(p1_vars, p2_vars):
        is_cat = (p1_name in cat_vars_set)

        if not is_cat:
            # Numeric Processing (Z-score)
            p1_raw = df[p1_name].values.astype(float)
            p2_raw = df[p2_name].values.astype(float)
            p1_m = (~np.isnan(p1_raw)).astype(int)
            p2_m = (~np.isnan(p2_raw)).astype(int)

            p2_obs = p2_raw[p2_m == 1]
            mu = np.mean(p2_obs) if len(p2_obs) > 0 else 0.0
            sigma = np.std(p2_obs) if len(p2_obs) > 0 else 1.0
            if sigma < 1e-6: sigma = 1.0

            normalization_stats[p2_name] = {
                'type': 'numeric',
                'mu': mu,
                'sigma': sigma
            }

            p1_final = np.nan_to_num((p1_raw - mu) / sigma, nan=0.0).reshape(-1, 1) * p1_m.reshape(-1, 1)
            p2_final = np.nan_to_num((p2_raw - mu) / sigma, nan=0.0).reshape(-1, 1) * p2_m.reshape(-1, 1)

            processed_data_list.extend([p1_final, p2_final])
            processed_mask_list.extend([p1_m.reshape(-1, 1), p2_m.reshape(-1, 1)])
            p1_generated_names.append(p1_name)

            p1_indices.append(current_col_idx)
            p2_indices.append(current_col_idx + 1)
            cat_p2_groups.append(None)
            current_col_idx += 2

        else:
            # Categorical Processing (Keep 0/1)
            s1_clean = get_unified_string_series(df[p1_name])
            s2_clean = get_unified_string_series(df[p2_name])
            all_cats = sorted(list(set(s1_clean.dropna().unique()) | set(s2_clean.dropna().unique())))
            n_new_cols = len(all_cats)

            p1_cat = pd.Categorical(s1_clean, categories=all_cats)
            p2_cat = pd.Categorical(s2_clean, categories=all_cats)

            p1_dum = pd.get_dummies(p1_cat, dummy_na=False).values.astype(float)
            p2_dum = pd.get_dummies(p2_cat, dummy_na=False).values.astype(float)

            p1_m_vec = df[p1_name].notna().astype(int).values.reshape(-1, 1)
            p2_m_vec = df[p2_name].notna().astype(int).values.reshape(-1, 1)
            p1_mask_chunk = np.tile(p1_m_vec, (1, n_new_cols))
            p2_mask_chunk = np.tile(p2_m_vec, (1, n_new_cols))

            normalization_stats[p2_name] = {
                'type': 'categorical',
                'categories': all_cats
            }

            processed_data_list.extend([p1_dum, p2_dum])
            processed_mask_list.extend([p1_mask_chunk, p2_mask_chunk])

            for cat in all_cats:
                p1_generated_names.append(f"{p1_name}_{cat}")

            p1_indices.extend(list(range(current_col_idx, current_col_idx + n_new_cols)))
            p2_indices.extend(list(range(current_col_idx + n_new_cols, current_col_idx + 2 * n_new_cols)))
            cat_p2_groups.append(list(range(current_col_idx + n_new_cols, current_col_idx + 2 * n_new_cols)))
            current_col_idx += 2 * n_new_cols

    # Weight Processing
    weight_col_idx = None
    if weight_var and weight_var in df.columns:
        w_raw = df[weight_var].values.astype(float)
        w_mean = np.mean(w_raw)
        normalization_stats['__weight__'] = {'mean': w_mean}

        w_norm = np.nan_to_num((w_raw / w_mean).reshape(-1, 1), nan=0.0)
        w_mask = (~np.isnan(w_raw)).astype(int).reshape(-1, 1)

        processed_data_list.append(w_norm)
        processed_mask_list.append(w_mask)
        weight_col_idx = current_col_idx

    final_data = np.hstack(processed_data_list)
    final_mask = np.hstack(processed_mask_list)

    return final_data, final_mask, p1_indices, p2_indices, weight_col_idx, p1_generated_names, normalization_stats, cat_p2_groups


def inverse_transform_data(processed_data, normalization_stats, data_info):
    """
    Reconstructs original data.
    """
    p1_vars = data_info['phase1_vars']
    p2_vars = data_info['phase2_vars']
    weight_var = data_info.get('weight_var')

    reconstructed_cols = {}
    current_col_idx = 0

    for p1_name, p2_name in zip(p1_vars, p2_vars):
        stats = normalization_stats[p2_name]

        if stats['type'] == 'numeric':
            mu = stats['mu']
            sigma = stats['sigma']
            reconstructed_cols[p1_name] = processed_data[:, current_col_idx] * sigma + mu
            reconstructed_cols[p2_name] = processed_data[:, current_col_idx + 1] * sigma + mu
            current_col_idx += 2

        elif stats['type'] == 'categorical':
            categories = stats['categories']
            n_cats = len(categories)

            p1_chunk = processed_data[:, current_col_idx: current_col_idx + n_cats]
            p2_chunk = processed_data[:, current_col_idx + n_cats: current_col_idx + 2 * n_cats]

            p2_chunk = np.exp(p2_chunk - np.max(p2_chunk, axis=1, keepdims=True))
            p2_chunk /= np.sum(p2_chunk, axis=1, keepdims=True)

            p1_argmax = np.argmax(p1_chunk, axis=1)
            p2_argmax = np.argmax(p2_chunk, axis=1)

            cat_map = np.array(categories)
            reconstructed_cols[p1_name] = cat_map[p1_argmax]
            reconstructed_cols[p2_name] = cat_map[p2_argmax]

            current_col_idx += 2 * n_cats

    if weight_var and '__weight__' in normalization_stats:
        w_mean = normalization_stats['__weight__']['mean']
        reconstructed_cols[weight_var] = processed_data[:, current_col_idx] * w_mean

    return pd.DataFrame(reconstructed_cols)