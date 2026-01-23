# utils.py
import pandas as pd
import numpy as np
import math


def decimal_to_bits(values, num_bits):
    """
    Converts integer array to analog bits (floats -1.0 and 1.0).
    Input: Integer array (N,)
    Output: Float array (N, num_bits)
    """
    vals = values.astype(int)
    bits_list = []
    for i in range(num_bits):
        # Extract i-th bit: (vals >> i) & 1. Little Endian.
        bit_col = (vals >> i) & 1
        bits_list.append(bit_col)

    # Stack to (N, num_bits)
    bits_arr = np.stack(bits_list, axis=1)

    # Map {0, 1} -> {-1.0, 1.0} for standard Gaussian diffusion compatibility
    return (bits_arr * 2.0) - 1.0


def bits_to_decimal(bits_pred):
    """
    Converts analog bits back to integers.
    Threshold: x > 0 -> 1, x <= 0 -> 0
    """
    bits_binary = (bits_pred > 0).astype(int)
    N, num_bits = bits_binary.shape
    integers = np.zeros(N, dtype=int)
    for i in range(num_bits):
        integers += bits_binary[:, i] * (2 ** i)
    return integers


def process_data(filepath, data_info):
    """
    Encodes data for Analog Bit Diffusion.
    - Categorical: Converted to Analog Bits (-1, 1).
    - Numeric: Standard normalized.
    - Stats: Calculated on UNION of P1 and P2 to ensure shared manifold.
    - Rows: Filters for valid P2 rows for training.
    """
    df = pd.read_csv(filepath)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    p1_vars = data_info['phase1_vars']
    p2_vars = data_info['phase2_vars']

    if len(p1_vars) != len(p2_vars):
        raise ValueError("Phase 1 and Phase 2 variable lists must be of equal length.")

    processed_data_list = []
    variable_schema = []

    # Indices to help slicing the flattened data array later (start, end)
    p1_indices = []
    p2_indices = []

    normalization_stats = {}
    current_col_idx = 0

    # Track validity of P2 rows for training mask (intersection of all P2 cols)
    p2_validity_masks = []

    print(f"\n[Data Processing] Encoding with Analog Bits & Joint Statistics...")

    for p1_name, p2_name in zip(p1_vars, p2_vars):
        if p1_name not in df.columns or p2_name not in df.columns:
            raise ValueError(f"Missing columns: {p1_name} or {p2_name}")

        p1_raw = df[p1_name].values
        p2_raw = df[p2_name].values

        # Mask: 1 if present, 0 if NaN
        m1 = (~df[p1_name].isna()).values
        m2 = (~df[p2_name].isna()).values
        p2_validity_masks.append(m2)

        is_categorical = (p2_name in data_info.get('cat_vars', [])) or (p1_name in data_info.get('cat_vars', []))

        if is_categorical:
            # 1. Joint Vocabulary Construction (P1 Union P2)
            valid_p1 = p1_raw[m1]
            valid_p2 = p2_raw[m2]

            unique_set = set(pd.unique(valid_p1)) | set(pd.unique(valid_p2))
            master_categories = sorted(list(unique_set))

            cat_to_int = {val: i for i, val in enumerate(master_categories)}
            K = len(master_categories)
            if K == 0: K = 1

            # Calculate required bits: L = ceil(log2(K))
            num_bits = max(1, math.ceil(math.log2(K)))

            # Map strings -> Integers
            p1_ints = np.array([cat_to_int.get(x, 0) for x in p1_raw])
            p2_ints = np.array([cat_to_int.get(x, 0) for x in p2_raw])

            # Integers -> Analog Bits
            d1 = decimal_to_bits(p1_ints, num_bits)  # Shape (N, num_bits)
            d2 = decimal_to_bits(p2_ints, num_bits)  # Shape (N, num_bits)

            variable_schema.append({
                'name': p2_name,
                'type': 'bits',
                'dim': num_bits,  # Vector dimension
                'num_classes': K
            })
            normalization_stats[p2_name] = {'type': 'bits', 'categories': np.array(master_categories)}
            normalization_stats[p1_name] = normalization_stats[p2_name]

            width = num_bits

        else:
            # 2. Joint Numeric Normalization (P1 Union P2)
            valid_p1 = p1_raw[m1].astype(float)
            valid_p2 = p2_raw[m2].astype(float)

            combined_valid = np.concatenate([valid_p1, valid_p2])

            if len(combined_valid) > 0:
                mu = np.mean(combined_valid)
                sigma = np.std(combined_valid)
                if sigma < 1e-6: sigma = 1.0
            else:
                mu, sigma = 0.0, 1.0

            d1 = ((p1_raw.astype(float) - mu) / sigma).reshape(-1, 1)
            d2 = ((p2_raw.astype(float) - mu) / sigma).reshape(-1, 1)

            # Fill NaNs with 0 (Mean) for storage
            d1 = np.nan_to_num(d1, nan=0.0)
            d2 = np.nan_to_num(d2, nan=0.0)

            variable_schema.append({
                'name': p2_name,
                'type': 'numeric',
                'dim': 1
            })
            normalization_stats[p2_name] = {'type': 'numeric', 'mu': mu, 'sigma': sigma}
            normalization_stats[p1_name] = normalization_stats[p2_name]

            width = 1

        processed_data_list.extend([d1, d2])

        # Track indices in the flattened array
        # P1 is at [current, current+width], P2 is at [current+width, current+2*width]
        p1_indices.append((current_col_idx, current_col_idx + width))
        p2_indices.append((current_col_idx + width, current_col_idx + 2 * width))
        current_col_idx += 2 * width

    # Aux Weight Var
    weight_idx = None
    w_var = data_info.get('weight_var')
    if w_var:
        w_raw = df[w_var].values
        m_w = (~df[w_var].isna()).values
        valid_w = w_raw[m_w].astype(float)

        if len(valid_w) > 0:
            mu_w, sig_w = np.mean(valid_w), np.std(valid_w)
            if sig_w < 1e-6: sig_w = 1.0
        else:
            mu_w, sig_w = 0.0, 1.0

        d_w = np.nan_to_num((w_raw.astype(float) - mu_w) / sig_w, nan=0.0).reshape(-1, 1)
        processed_data_list.append(d_w)
        variable_schema.append({'name': w_var, 'type': 'numeric_aux', 'dim': 1})
        weight_idx = (current_col_idx, current_col_idx + 1)
        current_col_idx += 1

    final_data = np.hstack(processed_data_list)

    # 3. Identify valid training rows (where P2 is fully observed)
    if p2_validity_masks:
        all_p2_masks = np.column_stack(p2_validity_masks)
        row_is_valid = all_p2_masks.all(axis=1)
        train_indices = np.where(row_is_valid)[0]
        print(f"Training on {len(train_indices)} / {len(df)} rows where Phase 2 is fully observed.")
    else:
        train_indices = np.arange(len(df))

    return (final_data, train_indices, p1_indices, p2_indices, weight_idx, variable_schema, normalization_stats, df)


def inverse_transform_data(processed_data, normalization_stats, data_info):
    """
    Decodes continuous model outputs back to Dataframe.
    Handles Bit->Decimal conversion and Denormalization.
    """
    reconstructed_df = pd.DataFrame()
    p2_vars = data_info['phase2_vars']

    curr_ptr = 0

    for p2_name in p2_vars:
        stats = normalization_stats[p2_name]

        if stats['type'] == 'bits':
            categories = stats['categories']
            num_cats = len(categories)
            num_bits = max(1, math.ceil(math.log2(num_cats)))

            # Slice bit vector
            bits_data = processed_data[:, curr_ptr: curr_ptr + num_bits]
            curr_ptr += num_bits

            # Decode bits -> int (Thresholding happens here)
            int_indices = bits_to_decimal(bits_data)

            # Clip to valid range (Model might dream up index K+1)
            int_indices = np.clip(int_indices, 0, num_cats - 1)

            reconstructed_df[p2_name] = categories[int_indices]

        elif stats['type'] == 'numeric':
            col_data = processed_data[:, curr_ptr]
            curr_ptr += 1

            mu, sigma = stats['mu'], stats['sigma']
            reconstructed_df[p2_name] = col_data * sigma + mu

    return reconstructed_df