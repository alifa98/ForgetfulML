import numpy as np

def engagement_score(r: np.ndarray, f: np.ndarray) -> float:
    """
    Compute the "engagement score" between a retain set and a forget set, both 
    being collections of d-dimensional observations.

    The score compares the average intra-group variance within each set (retain and forget)
    to the variance between their means and the combined mean, providing a measure 
    of how distinct or separable these two groups are.

    Parameters
    ----------
    r : np.ndarray
        A numpy array of shape (n, d), representing the retain set with n observations 
        each of dimension d.
    f : np.ndarray
        A numpy array of shape (m, d), representing the forget set with m observations 
        each of dimension d.

    Returns
    -------
    float
        The engagement score, a non-negative number that reflects the relationship
        between intra-group variance and inter-group variance.

    Raises
    ------
    ValueError
        If the denominator is zero, which means the means of r and f coincide with 
        the combined mean, making the score undefined.
    """
    mu_r = r.mean(axis=0)  # shape (d,)
    mu_f = f.mean(axis=0)  # shape (d,)

    combined = np.vstack((r, f))
    mu_combined = combined.mean(axis=0)  # shape (d,)

    # Compute numerator (sum of intra-group variances)
    diff_r = r - mu_r  # shape (n, d)
    diff_f = f - mu_f  # shape (m, d)

    numerator_part_r = np.sum(diff_r**2) / r.shape[0]
    numerator_part_f = np.sum(diff_f**2) / f.shape[0]
    numerator = numerator_part_r + numerator_part_f

    # Compute denominator (variance of the group means around the combined mean)
    diff_mu_r = mu_r - mu_combined
    diff_mu_f = mu_f - mu_combined
    denominator = 0.5 * (np.sum(diff_mu_r**2) + np.sum(diff_mu_f**2))

    if denominator == 0:
        raise ValueError("Denominator is zero, cannot compute engagement score.")

    es_value = numerator / denominator
    return float(es_value)
