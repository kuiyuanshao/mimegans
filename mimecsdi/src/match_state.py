# match_state.py: Estimates diffusion steps for each variable using bootstrap matching on audit data.
import numpy as np
from tqdm import tqdm
from scipy.stats import norm

def get_diffusion_schedule(config):
    start = config["diffusion"]["beta_start"]
    end = config["diffusion"]["beta_end"]
    num_steps = config["diffusion"]["num_steps"]
    schedule = config["diffusion"]["schedule"]

    if schedule == "linear":
        betas = np.linspace(start, end, num_steps, dtype=np.float64)
    elif schedule == "quad":
        betas = np.linspace(start ** 0.5, end ** 0.5, num_steps, dtype=np.float64) ** 2

    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas)
    return alphas_cumprod

def match_state(processed_values, processed_mask,
                p1_indices, p2_indices,
                col_names, weight_idx=None, config=None, n_bootstrap=500, normalization_stats=None):
    """
    Estimates diffusion steps using dual-track matching:
    - Numeric: Residual standard deviation matching (Z-score space).
    - Binary: Disagreement rate matching via Gaussian CDF (0/1 space).
    """

    alphas_cumprod = get_diffusion_schedule(config)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas = np.sqrt(1. - alphas_cumprod)

    theoretical_flip_probs = norm.cdf(-0.5 / sqrt_one_minus_alphas)

    try:
        observed_values = np.array(processed_values, dtype=float)
    except ValueError:
        raise ValueError("Data must be float.")

    if len(p2_indices) > 0:
        audit_indices = np.where(processed_mask[:, p2_indices[0]] == 1)[0]
    else:
        return np.array([]), []

    n_audit = len(audit_indices)
    audit_data = observed_values[audit_indices]

    if weight_idx is not None:
        weights = observed_values[audit_indices, weight_idx]
        weights = weights / np.mean(weights)
    else:
        weights = np.ones(n_audit)

    n_cols = len(p1_indices)
    matched_steps_matrix = np.zeros((n_cols, n_bootstrap), dtype=int)
    is_binary = [False] * n_cols

    p2_var_names = []
    for i in range(n_cols):
        name = col_names[i] if col_names else f"Col_{i}"
        base_name = name.split('_')[0] if '_' in name else name
        p2_var_names.append(base_name)

    for i in range(n_cols):
        base_name = p2_var_names[i]
        stats = normalization_stats.get(base_name, {})
        if stats.get('type') == 'categorical':
            is_binary[i] = True
        else:
            unique_vals = np.unique(audit_data[:, p2_indices[i]])
            if len(unique_vals) <= 2 and np.all(unique_vals == np.round(unique_vals)):
                is_binary[i] = True

    print(f"Starting Dual-Track State Matching (Binary 0/1 & Numeric Z-score) for {n_cols} columns...")

    for b in tqdm(range(n_bootstrap), desc="Bootstrapping"):
        p_sample = weights / np.sum(weights)
        resample_idx = np.random.choice(n_audit, size=n_audit, replace=True, p=p_sample)

        b_data = audit_data[resample_idx]
        b_w = weights[resample_idx]
        w_sum = np.sum(b_w)

        y_proxy = b_data[:, p1_indices]
        y_true = b_data[:, p2_indices]

        for i in range(n_cols):
            if is_binary[i]:
                disagreement = ((y_proxy[:, i] > 0.5) != (y_true[:, i] > 0.5)).astype(float)
                emp_error_rate = np.sum(disagreement * b_w) / w_sum

                diffs = np.abs(theoretical_flip_probs - emp_error_rate)
                matched_steps_matrix[i, b] = np.argmin(diffs)

            else:
                res_col = y_proxy[:, i:i + 1] - y_true[:, i:i + 1] * sqrt_alphas_cumprod.reshape(1, -1)

                mean_res = np.sum(res_col * b_w.reshape(-1, 1), axis=0) / w_sum
                centered_sq = (res_col - mean_res) ** 2
                var_res = np.sum(centered_sq * b_w.reshape(-1, 1), axis=0) / w_sum
                std_res = np.sqrt(var_res)

                diffs = np.abs(std_res - sqrt_one_minus_alphas)
                matched_steps_matrix[i, b] = np.argmin(diffs)

    final_matched_steps = np.median(matched_steps_matrix, axis=1).astype(int)

    print("\n" + "=" * 65)
    print(f"{'Variable / Level':<35} | {'Type':<10} | {'Step':<10}")
    print("-" * 65)

    for i in range(n_cols):
        name_str = col_names[i] if (col_names and i < len(col_names)) else f"Col_{p1_indices[i]}"
        type_str = "Binary" if is_binary[i] else "Numeric"
        print(f"{name_str:<35} | {type_str:<10} | {final_matched_steps[i]:<10}")

    print("=" * 65 + "\n")

    return final_matched_steps, is_binary