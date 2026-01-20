# match_state.py: Estimates diffusion steps using sequential Analog-consistent Variance Matching.
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import sys


def get_diffusion_schedule(config):
    """
    Returns the cumulative product of alphas (alpha_bar) based on noise schedule.
    """
    start = config["diffusion"]["beta_start"]
    end = config["diffusion"]["beta_end"]
    num_steps = config["diffusion"]["num_steps"]
    schedule = config["diffusion"]["schedule"]

    if schedule == "linear":
        betas = np.linspace(start, end, num_steps, dtype=np.float64)
    elif schedule == "quad":
        # Quadratic schedule: standard for Bit Diffusion / Analog Bits
        betas = np.linspace(start ** 0.5, end ** 0.5, num_steps, dtype=np.float64) ** 2
    else:
        betas = np.linspace(start, end, num_steps, dtype=np.float64)

    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas)
    return alphas_cumprod


def match_state(processed_values, processed_mask, p1_indices, p2_indices, col_names,
                weight_idx=None, config=None, n_bootstrap=500, normalization_stats=None, data_info=None):
    """
    State Matching using Sequential Search and Early Stopping.
    Treats both bits and numeric as continuous analog signals.
    """
    # 1. Setup Schedules
    alphas_cumprod = get_diffusion_schedule(config)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)

    # Theoretical noise scale: sqrt(1 - alpha_bar)
    # Note: user's snippet uses sqrt(1 - sqrt_alpha**2), which is sqrt(1 - alpha)
    theoretical_std = np.sqrt(1. - alphas_cumprod)

    n_rows, n_cols = processed_values.shape[0], len(p2_indices)
    weights = processed_values[:, weight_idx] if weight_idx is not None else np.ones(n_rows)

    # 2. Extract Audit Set (where Phase-2 is ground-truth)
    audit_mask = processed_mask[:, p2_indices[0]] == 1
    audit_data = processed_values[audit_mask]
    audit_weights = weights[audit_mask]
    w_sum = np.sum(audit_weights)

    # 3. Identify Bit Channels (for final report only)
    is_bit = np.zeros(n_cols, dtype=bool)
    curr = 0
    for p2_name in data_info['phase2_vars']:
        stat = normalization_stats[p2_name]
        if stat['type'] == 'numeric':
            is_bit[curr] = False
            curr += 1
        else:
            num_bits = stat['num_bits']
            is_bit[curr: curr + num_bits] = True
            curr += num_bits

    matched_steps_matrix = np.zeros((n_cols, n_bootstrap))

    print(f"   [Matching] Analyzing {n_cols} channels using Sequential Search ({n_bootstrap} iterations)...")

    # 4. Bootstrap Loop for PhD-level statistical robustness
    for b in tqdm(range(n_bootstrap), desc="   Matching Progress", file=sys.stdout):
        # Bootstrap sampling
        idx = np.random.choice(len(audit_data), len(audit_data), replace=True)
        b_data = audit_data[idx]
        b_w = audit_weights[idx]
        b_w_sum = np.sum(b_w)

        for i in range(n_cols):
            # Sequential search logic per your provided snippet
            curr_p1 = b_data[:, p1_indices[i]]
            curr_p2 = b_data[:, p2_indices[i]]

            best_t = 0
            best_diff = 999.
            prev_diff = 999.

            # Search from t=0 to T
            for t in range(len(sqrt_alphas_cumprod)):
                # Calculate residual: noise = P1 - sqrt(alpha)*P2
                noise = curr_p1 - sqrt_alphas_cumprod[t] * curr_p2

                # Weighted mean and std calculation (matching norm.fit logic but with weights)
                noise_mean = np.sum(noise * b_w) / b_w_sum
                centered_noise = noise - noise_mean
                noise_std = np.sqrt(np.sum((centered_noise ** 2) * b_w) / b_w_sum)

                # diff = |theoretical_noise_scale - empirical_noise_std|
                diff = np.abs(theoretical_std[t] - noise_std)

                # Early stopping logic from your original snippet
                if diff < best_diff:
                    best_diff = diff
                    best_t = t

                if diff > prev_diff:
                    # Found a local match!
                    break
                else:
                    prev_diff = diff

            matched_steps_matrix[i, b] = best_t

    # 5. Aggregate results using median
    final_matched_steps = np.median(matched_steps_matrix, axis=1).astype(int)

    # 6. Final Convenience Report
    print("\n" + "=" * 50)
    print(f"{'Channel':<10} | {'Type':<6} | {'Matched t':<10}")
    print("-" * 50)
    for i, t_val in enumerate(final_matched_steps):
        v_type = "BIT" if is_bit[i] else "NUM"
        print(f"{i:<10} | {v_type:<6} | {t_val:<10}")
    print("=" * 50 + "\n")

    return final_matched_steps, is_bit