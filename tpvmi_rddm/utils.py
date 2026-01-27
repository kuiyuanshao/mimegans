import pandas as pd
import numpy as np


def process_data(filepath, data_info):
    """
    Reads data and encodes variables for the Unified Diffusion Model.

    Returns:
        final_data: Numpy array [P1_0, P2_0, ..., Aux_0, ...]
        variable_schema: List of dicts describing P2 targets and Aux variables.
        normalization_stats: Dict of mu/sigma/categories for inversion.
    """
    df = pd.read_csv(filepath)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    p1_vars = data_info.get('phase1_vars', [])
    p2_vars = data_info.get('phase2_vars', [])
    cat_vars_set = set(data_info.get('cat_vars', []))

    if len(p1_vars) != len(p2_vars):
        raise ValueError("Phase 1 and Phase 2 variable lists must be of equal length.")

    # [1] Identify Auxiliaries (Context)
    # Any variable that is NOT a P1 or P2 target is an Auxiliary.
    reserved_vars = set(p1_vars) | set(p2_vars)
    all_vars = df.columns.tolist()
    aux_vars = sorted([c for c in all_vars if c not in reserved_vars])

    processed_data_list = []
    processed_mask_list = []
    variable_schema = []

    # Indices to help TPVMI_RDDM slice the tensors later
    p1_indices = []
    p2_indices = []

    current_col_idx = 0
    normalization_stats = {}
    weight_idx = None  # Legacy compatibility

    print(f"\n[Data Processing] Targets: {len(p2_vars)} pairs | Context: {len(aux_vars)} aux variables")

    # ==========================================
    # BLOCK A: PHASE 1 & PHASE 2 (Target Pairs)
    # ==========================================
    for p1_name, p2_name in zip(p1_vars, p2_vars):
        if p1_name not in df.columns or p2_name not in df.columns:
            raise ValueError(f"Missing pair: {p1_name} or {p2_name}")

        p1_raw = df[p1_name].values
        p2_raw = df[p2_name].values

        # Masks: 1=Observed, 0=Missing
        m1 = (~df[p1_name].isna()).values.astype(float)
        m2 = (~df[p2_name].isna()).values.astype(float)

        is_categorical = (p2_name in cat_vars_set) or (p1_name in cat_vars_set)

        if is_categorical:
            # --- CATEGORICAL ---
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

            # Schema: Only P2 is added as a 'target' to be generated.
            variable_schema.append({'name': p2_name, 'type': 'categorical', 'num_classes': K})
            normalization_stats[p2_name] = {'type': 'categorical', 'categories': np.array(master_categories)}
            normalization_stats[p1_name] = normalization_stats[p2_name]

        else:
            # --- NUMERIC ---
            v1_float = p1_raw.astype(float)
            v2_float = p2_raw.astype(float)

            combined_min = np.nanmin(np.concatenate([v1_float, v2_float]))
            shift = 0.0
            if combined_min < 0:
                shift = -combined_min

            # v1_float = np.log1p(v1_float + shift)
            # v2_float = np.log1p(v2_float + shift)

            # Norm Stats based on Observed P2
            valid_p2_log = v2_float[m2 == 1]
            if len(valid_p2_log) > 0:
                mu, sigma = np.mean(valid_p2_log), np.std(valid_p2_log)
                if sigma < 1e-6: sigma = 1.0
            else:
                mu, sigma = 0.0, 1.0

            d1 = (v1_float - mu) / sigma
            d2 = (v2_float - mu) / sigma

            d1 = np.nan_to_num(d1, nan=0.0).reshape(-1, 1)
            d2 = np.nan_to_num(d2, nan=0.0).reshape(-1, 1)

            processed_data_list.extend([d1, d2])
            processed_mask_list.extend([m1.reshape(-1, 1), m2.reshape(-1, 1)])

            variable_schema.append({'name': p2_name, 'type': 'numeric'})
            normalization_stats[p2_name] = {'type': 'numeric', 'mu': mu, 'sigma': sigma, 'shift': shift}
            normalization_stats[p1_name] = normalization_stats[p2_name]

        # P1 is at current, P2 is at current+1
        p1_indices.append(current_col_idx)
        p2_indices.append(current_col_idx + 1)
        current_col_idx += 2

    # ==========================================
    # BLOCK B: AUXILIARY CONTEXT
    # ==========================================
    for aux_name in aux_vars:
        raw_vals = df[aux_name].values
        mask = (~df[aux_name].isna()).values.astype(float).reshape(-1, 1)

        is_categorical = aux_name in cat_vars_set

        if is_categorical:
            u_vals = pd.unique(raw_vals[pd.notna(raw_vals)])
            master_categories = sorted(list(u_vals))
            cat_to_int = {val: i for i, val in enumerate(master_categories)}
            K = len(master_categories)
            if K == 0: K = 1

            d_aux = np.array([cat_to_int.get(x, 0) for x in raw_vals]).reshape(-1, 1)

            variable_schema.append({'name': aux_name, 'type': 'categorical_aux', 'num_classes': K})
            normalization_stats[aux_name] = {'type': 'categorical', 'categories': np.array(master_categories)}

        else:
            v_float = raw_vals.astype(float)
            v_min = np.nanmin(v_float)
            shift = 0.0
            if v_min < 0: shift = -v_min

            valid_log = v_float[mask.flatten() == 1]
            if len(valid_log) > 0:
                mu, sigma = np.mean(valid_log), np.std(valid_log)
                if sigma < 1e-6: sigma = 1.0
            else:
                mu, sigma = 0.0, 1.0

            d_aux = (v_float - mu) / sigma
            d_aux = np.nan_to_num(d_aux, nan=0.0).reshape(-1, 1)

            variable_schema.append({'name': aux_name, 'type': 'numeric_aux'})
            normalization_stats[aux_name] = {'type': 'numeric', 'mu': mu, 'sigma': sigma, 'shift': shift}

        processed_data_list.append(d_aux)
        processed_mask_list.append(mask)
        current_col_idx += 1

    final_data = np.hstack(processed_data_list)
    final_mask = np.hstack(processed_mask_list)

    return (final_data, final_mask, np.array(p1_indices), np.array(p2_indices),
            weight_idx, variable_schema, normalization_stats, df)


def inverse_transform_data(generated_data, normalization_stats, data_info):
    """
    Reverses normalization for the GENERATED data.

    [CRITICAL FIX]:
    - The input 'generated_data' comes from impute(), which contains ONLY the P2 columns.
    - It has shape (N, Num_P2_Vars).
    - We iterate through 'phase2_vars' and map them 1-to-1 with the columns of generated_data.
    """
    reconstructed_df = pd.DataFrame()
    p2_vars = data_info['phase2_vars']

    # Check for shape consistency
    if generated_data.shape[1] != len(p2_vars):
        # Fallback logic if the input happens to be the full matrix (unlikely given tpvmi_rddm.py logic)
        print(f"Warning: generated_data shape {generated_data.shape} does not match p2_vars count {len(p2_vars)}.")

    for i, p2_name in enumerate(p2_vars):
        stats = normalization_stats[p2_name]

        # Since generated_data ONLY contains P2 results, we access column 'i' directly.
        col_data = generated_data[:, i]

        if stats['type'] == 'numeric':
            mu, sigma, shift = stats['mu'], stats['sigma'], stats['shift']
            # Reverse Z-score
            final_val = col_data * sigma + mu
            reconstructed_df[p2_name] = final_val
        else:
            categories = stats['categories']
            indices = np.clip(np.round(col_data), 0, len(categories) - 1).astype(int)
            if len(categories) > 0:
                reconstructed_df[p2_name] = categories[indices]
            else:
                reconstructed_df[p2_name] = indices

    return reconstructed_df