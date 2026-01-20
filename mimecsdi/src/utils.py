# utils.py: Data processing utilities for reading, cleaning, and Analog Bits encoding/decoding.
import pandas as pd
import numpy as np
import math


def process_data(filepath, data_info):
    """
    Reads data, cleans, and encodes variables.
    - Numeric features: Z-score normalization.
    - Categorical features: Binary bit encoding (Analog Bits) scaled to [-1, 1].
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
    cat_p2_groups = []

    normalization_stats = {}
    current_col_idx = 0

    # Helper function to clean and unify string categories
    def get_unified_string_series(s):
        s_num = pd.to_numeric(s, errors='coerce')
        if s_num.notna().any():
            non_na = s_num.dropna()
            if (non_na % 1 == 0).all():
                return s_num.astype('Int64').astype(str).replace('<NA>', np.nan)
        return s.astype(str).str.strip().replace('nan', np.nan)

    for p1_name, p2_name in zip(p1_vars, p2_vars):
        if p2_name not in cat_vars_set:
            # --- Numeric Variable Treatment (Standard Z-score) ---
            combined = pd.concat([df[p1_name], df[p2_name]]).dropna()
            mu = combined.mean()
            sigma = combined.std() + 1e-6

            normalization_stats[p2_name] = {'type': 'numeric', 'mu': mu, 'sigma': sigma}

            p1_val = ((df[p1_name] - mu) / sigma).fillna(0).values[:, None]
            p2_val = ((df[p2_name] - mu) / sigma).fillna(0).values[:, None]

            p1_mask = df[p1_name].notna().astype(float).values[:, None]
            p2_mask = df[p2_name].notna().astype(float).values[:, None]

            p1_indices.append(current_col_idx)
            p2_indices.append(current_col_idx + 1)

            processed_data_list.extend([p1_val, p2_val])
            processed_mask_list.extend([p1_mask, p2_mask])
            current_col_idx += 2

        else:
            # --- Analog Bits Treatment: Binary Bit Encoding (ICLR 2023) ---
            s1 = get_unified_string_series(df[p1_name])
            s2 = get_unified_string_series(df[p2_name])
            combined_cats = sorted(list(pd.concat([s1, s2]).dropna().unique()))
            n_cats = len(combined_cats)

            # Determine bits required: n = ceil(log2(K))
            num_bits = math.ceil(math.log2(n_cats)) if n_cats > 1 else 1

            normalization_stats[p2_name] = {
                'type': 'categorical',
                'categories': combined_cats,
                'num_bits': num_bits,
                'cat_to_id': {cat: i for i, cat in enumerate(combined_cats)}
            }

            def encode_bits(series, n_bits, cat_map):
                ids = series.map(cat_map).fillna(0).astype(int).values
                # Extract bit-planes (from MSB to LSB)
                bit_matrix = np.array([(ids >> i) & 1 for i in range(n_bits)[::-1]]).T
                # Scale from [0, 1] to [-1, 1] for Analog Bits compatibility
                return bit_matrix.astype(float) * 2.0 - 1.0

            p1_bits = encode_bits(s1, num_bits, normalization_stats[p2_name]['cat_to_id'])
            p2_bits = encode_bits(s2, num_bits, normalization_stats[p2_name]['cat_to_id'])

            p1_mask = np.repeat(df[p1_name].notna().astype(float).values[:, None], num_bits, axis=1)
            p2_mask = np.repeat(df[p2_name].notna().astype(float).values[:, None], num_bits, axis=1)

            p1_indices.extend(range(current_col_idx, current_col_idx + num_bits))
            p2_indices.extend(range(current_col_idx + num_bits, current_col_idx + 2 * num_bits))
            cat_p2_groups.append(list(range(current_col_idx + num_bits, current_col_idx + 2 * num_bits)))

            processed_data_list.extend([p1_bits, p2_bits])
            processed_mask_list.extend([m for m in [p1_mask, p2_mask]])
            current_col_idx += 2 * num_bits

    final_data = np.concatenate(processed_data_list, axis=1)
    final_mask = np.concatenate(processed_mask_list, axis=1)

    # Weight Handling
    weight_idx = final_data.shape[1] if weight_var and weight_var in df.columns else None
    if weight_idx:
        w_val = df[weight_var].fillna(df[weight_var].mean()).values[:, None]
        final_data = np.concatenate([final_data, w_val], axis=1)
        final_mask = np.concatenate([final_mask, np.ones((len(df), 1))], axis=1)

    return (final_data, final_mask, np.array(p1_indices), np.array(p2_indices),
            weight_idx, None, normalization_stats, cat_p2_groups)


def inverse_transform_data(processed_data, normalization_stats, data_info):
    """
    Decodes analog bits and normalized numeric data back to original form.
    """
    reconstructed_cols = {}
    current_col_idx = 0

    for p1_name, p2_name in zip(data_info['phase1_vars'], data_info['phase2_vars']):
        stats = normalization_stats[p2_name]

        if stats['type'] == 'numeric':
            mu, sigma = stats['mu'], stats['sigma']
            reconstructed_cols[p1_name] = processed_data[:, current_col_idx] * sigma + mu
            reconstructed_cols[p2_name] = processed_data[:, current_col_idx + 1] * sigma + mu
            current_col_idx += 2

        else:
            num_bits = stats['num_bits']
            categories = stats['categories']

            def decode_bits(chunk):
                # Threshold analog bits at 0.0 to recover discrete bits
                bits = (chunk > 0).astype(int)
                # Convert bits back to category ID
                powers = 1 << np.arange(num_bits)[::-1]
                ids = np.sum(bits * powers, axis=1)
                return np.array(categories)[np.clip(ids, 0, len(categories) - 1)]

            reconstructed_cols[p1_name] = decode_bits(processed_data[:, current_col_idx: current_col_idx + num_bits])
            reconstructed_cols[p2_name] = decode_bits(
                processed_data[:, current_col_idx + num_bits: current_col_idx + 2 * num_bits])
            current_col_idx += 2 * num_bits

    return pd.DataFrame(reconstructed_cols)