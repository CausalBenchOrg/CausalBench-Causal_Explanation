import itertools

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay

from common.common_constants import RANDOM_SEED


def _number_subspaces(n_dims):
    if n_dims % 3 == 0:
        return n_dims // 3, 0
    if n_dims % 3 == 1:
        return (n_dims - 4) // 3, 2
    return (n_dims - 2) // 3, 1


def _create_subspaces(n_dims):
    if 0 < n_dims <= 3:
        return [list(range(1, n_dims + 1))]
    if n_dims <= 0:
        raise ValueError("n_dims must be positive")

    n_sub3, n_sub2 = _number_subspaces(n_dims)
    splits = []
    for i in range(1, n_sub3 + 1):
        splits.append([3 * i - 2, 3 * i - 1, 3 * i])

    if n_sub2 == 1:
        splits.append([3 * n_sub3 + 1, 3 * n_sub3 + 2])
    elif n_sub2 == 2:
        splits.append([3 * n_sub3 + 1, 3 * n_sub3 + 2])
        splits.append([3 * n_sub3 + 3, 3 * n_sub3 + 4])

    return splits


def _initialize_splits(splits, discret):
    ndims_run = []
    discrets_run = []
    dims_left_run = []

    for i, split in enumerate(splits):
        split_idx = [s - 1 for s in split]
        discrets_run.append([discret[k] for k in split_idx])
        ndims_run.append(len(split_idx))

        before = sorted([d for ii, sp in enumerate(splits) if ii < i for d in sp])
        after = sorted([d for ii, sp in enumerate(splits) if ii > i for d in sp])
        dims_left_run.append(before + after)

    return ndims_run, discrets_run, dims_left_run


def _get_split_information(splits, ndims_run, discrets_run, dims_left_run, i):
    return (
        splits[i],
        ndims_run[i],
        discrets_run[i],
        dims_left_run[i],
    )


def _choose_num_centres(weights, new_budget, cov_t=0.9, alpha=2.0):
    w_sorted = np.sort(weights)[::-1]
    k_cov = int(np.searchsorted(np.cumsum(w_sorted), cov_t) + 1)
    return int(min(int(np.ceil(new_budget / alpha)), k_cov))


def _solid_angle_triangle(point, va, vb, vc):
    r_pa = point - va
    r_pb = point - vb
    r_pc = point - vc

    numerator = float(np.dot(r_pa, np.cross(r_pb, r_pc)))
    denominator = (
        np.linalg.norm(r_pa) * np.linalg.norm(r_pb) * np.linalg.norm(r_pc)
        + np.dot(r_pa, r_pb) * np.linalg.norm(r_pc)
        + np.dot(r_pa, r_pc) * np.linalg.norm(r_pb)
        + np.dot(r_pb, r_pc) * np.linalg.norm(r_pa)
    )
    return 2.0 * np.arctan2(numerator, denominator)


def _estimate_gradient(tri, samples, criticality=None):
    n_dims = samples.shape[1] - 1

    if criticality is None:
        values = samples[:, n_dims]
    else:
        values = np.asarray(criticality, dtype=float)

    simplices = tri.simplices
    n_simplices = simplices.shape[0]

    if n_dims == 2:
        i = samples[simplices[:, 0], :2]
        j = samples[simplices[:, 1], :2]
        k = samples[simplices[:, 2], :2]
        i_f = values[simplices[:, 0]]
        j_f = values[simplices[:, 1]]
        k_f = values[simplices[:, 2]]

        ik = i - k
        ji = j - i
        areas = 0.5 * np.abs((j[:, 0] - i[:, 0]) * (k[:, 1] - i[:, 1]) - (k[:, 0] - i[:, 0]) * (j[:, 1] - i[:, 1]))
        areas = np.maximum(areas, 1e-12)

        rot = np.array([[0, -1], [1, 0]], dtype=float)
        ikr = ik @ rot.T
        jir = ji @ rot.T
        grad = (j_f - i_f)[:, None] * (ikr / (2 * areas[:, None])) + (k_f - i_f)[:, None] * (jir / (2 * areas[:, None]))
    elif n_dims == 3:
        grad = np.zeros((n_simplices, 3), dtype=float)
        i = samples[simplices[:, 0], :3]
        j = samples[simplices[:, 1], :3]
        k = samples[simplices[:, 2], :3]
        h = samples[simplices[:, 3], :3]

        i_f = values[simplices[:, 0]]
        j_f = values[simplices[:, 1]]
        k_f = values[simplices[:, 2]]
        h_f = values[simplices[:, 3]]

        ji = j - i
        ki = k - i
        hi = h - i
        ik = i - k
        hk = h - k
        ih = i - h
        jh = j - h

        ji_f = j_f - i_f
        ki_f = k_f - i_f
        hi_f = h_f - i_f

        for t in range(n_simplices):
            vol = (1.0 / 6.0) * abs(np.linalg.det(np.vstack([ji[t], ki[t], hi[t]])))
            if vol > 0:
                grad[t] = (
                    ji_f[t] * (np.cross(ik[t], hk[t]) / (2 * vol))
                    + ki_f[t] * (np.cross(ih[t], jh[t]) / (2 * vol))
                    + hi_f[t] * (np.cross(ki[t], ji[t]) / (2 * vol))
                )
    else:
        raise ValueError("Only 2D and 3D gradients are supported")

    n_points = tri.points.shape[0]
    attachments = [[] for _ in range(n_points)]
    for tri_id, simplex in enumerate(simplices):
        for vertex in simplex:
            attachments[vertex].append(tri_id)

    grad_v = np.zeros((n_points, n_dims), dtype=float)
    points = tri.points

    if n_dims == 2:
        i = samples[simplices[:, 0], :2]
        j = samples[simplices[:, 1], :2]
        k = samples[simplices[:, 2], :2]
        areas = 0.5 * np.abs((j[:, 0] - i[:, 0]) * (k[:, 1] - i[:, 1]) - (k[:, 0] - i[:, 0]) * (j[:, 1] - i[:, 1]))
        for point_idx in range(n_points):
            tri_ids = attachments[point_idx]
            if not tri_ids:
                continue

            angles = []
            weighted_grad = np.zeros(n_dims)
            for tri_id in tri_ids:
                ids = simplices[tri_id].tolist()
                ids.remove(point_idx)
                p1, p2, p3 = points[point_idx], points[ids[0]], points[ids[1]]
                angle = np.arctan2(2 * areas[tri_id], np.dot(p2 - p1, p3 - p1))
                angles.append(angle)
                weighted_grad += angle * grad[tri_id]

            grad_v[point_idx] = weighted_grad / max(np.sum(angles), 1e-12)
    else:
        for point_idx in range(n_points):
            tri_ids = attachments[point_idx]
            if not tri_ids:
                continue

            angles = []
            weighted_grad = np.zeros(n_dims)
            for tri_id in tri_ids:
                ids = simplices[tri_id].tolist()
                ids.remove(point_idx)
                p1 = points[point_idx]
                p2, p3, p4 = points[ids[0]], points[ids[1]], points[ids[2]]
                angle = abs(_solid_angle_triangle(p1, p2, p3, p4))
                angles.append(angle)
                weighted_grad += angle * grad[tri_id]

            solid_angle = np.sum(angles)
            grad_value = weighted_grad / max(solid_angle, 1e-12)
            grad_v[point_idx] = grad_value / 2.0 if solid_angle < 4 * np.pi else grad_value

    return grad_v


def _propose_samples_new4(samples_tri, values, new_budget, discret_spl, rng, dim_weights=None, causal_mode=0):
    t_mix = 1.0
    sigma_fac = 0.02
    edge_cut = 0.025

    _, n_dims = samples_tri.shape
    if n_dims not in (2, 3):
        raise ValueError("Sampler is designed for 2D/3D subspaces")
    if causal_mode not in (0, 1, 2):
        raise ValueError("causal_mode must be 0 (both), 1 (proposal), or 2 (norm)")

    lower_bounds = np.array([values_arr[0] for values_arr in discret_spl], dtype=float)
    upper_bounds = np.array([values_arr[-1] for values_arr in discret_spl], dtype=float)

    if dim_weights is None:
        local_dim_weights = np.ones(n_dims, dtype=float)
    else:
        local_dim_weights = np.asarray(dim_weights, dtype=float).reshape(-1)
        if local_dim_weights.size != n_dims:
            raise ValueError("dim_weights must match the current subspace dimensionality")
        if np.any(local_dim_weights <= 0):
            raise ValueError("dim_weights must be strictly positive")
        local_dim_weights = local_dim_weights / np.mean(local_dim_weights)

    apply_weighted_norm = dim_weights is not None and causal_mode in (0, 2)
    apply_anisotropic_proposal = dim_weights is not None and causal_mode in (0, 1)

    if values.ndim == 2 and values.shape[1] >= 2:
        if apply_weighted_norm:
            values = np.sqrt(np.sum((values * local_dim_weights) ** 2, axis=1))
        else:
            values = np.linalg.norm(values, axis=1)

    x_norm = (samples_tri - lower_bounds) / np.maximum(upper_bounds - lower_bounds, 1e-12)
    sigma = sigma_fac * np.sqrt(n_dims)

    weights = values - np.min(values)
    if np.allclose(weights, 0):
        weights = np.ones_like(weights)
    weights = weights / np.sum(weights)

    mask_core = np.all((x_norm > edge_cut) & (x_norm < (1.0 - edge_cut)), axis=1)
    if not np.any(mask_core):
        mask_core = np.ones(x_norm.shape[0], dtype=bool)

    x_core = x_norm[mask_core]
    w_core = weights[mask_core]
    w_core = w_core / np.sum(w_core)

    n_centers = _choose_num_centres(w_core, new_budget)
    n_centers = min(n_centers, x_core.shape[0])
    center_indices = rng.choice(x_core.shape[0], size=n_centers, replace=True, p=w_core)
    centers = x_core[center_indices]
    center_weights = w_core[center_indices]
    center_weights = center_weights / np.sum(center_weights)

    proposal_scale = local_dim_weights if apply_anisotropic_proposal else np.ones(n_dims, dtype=float)

    x_new = np.zeros((new_budget, n_dims), dtype=float)
    for i in range(new_budget):
        if rng.random() < t_mix:
            center_idx = rng.choice(len(center_weights), p=center_weights)
            x_candidate = centers[center_idx] + sigma * proposal_scale * rng.normal(size=n_dims)
            x_candidate = np.clip(x_candidate, 0.0, 1.0)
        else:
            x_candidate = rng.random(size=n_dims)
        x_new[i] = lower_bounds + x_candidate * (upper_bounds - lower_bounds)

    linear_interp = LinearNDInterpolator(samples_tri, values, fill_value=np.nan)
    vhat = linear_interp(x_new)
    missing = np.isnan(vhat)
    if np.any(missing):
        nearest_interp = NearestNDInterpolator(samples_tri, values)
        vhat[missing] = nearest_interp(x_new[missing])

    x_new_val = np.column_stack([x_new, vhat.reshape(-1)])
    return x_new, x_new_val


def _propose_1d_gradient_samples(samples_tri, values, discret_spl, new_budget, rng):
    if samples_tri.size == 0:
        return np.zeros((0, 1), dtype=float)

    coords = samples_tri.reshape(-1).astype(float)
    values = np.asarray(values, dtype=float).reshape(-1)

    order = np.argsort(coords)
    coords = coords[order]
    values = values[order]

    unique_coords, inverse = np.unique(coords, return_inverse=True)
    unique_values = np.zeros(unique_coords.shape[0], dtype=float)
    counts = np.zeros(unique_coords.shape[0], dtype=float)
    for idx, group_idx in enumerate(inverse):
        unique_values[group_idx] += values[idx]
        counts[group_idx] += 1
    unique_values = unique_values / np.maximum(counts, 1.0)

    if unique_coords.shape[0] < 2:
        return np.zeros((0, 1), dtype=float)

    left = unique_coords[:-1]
    right = unique_coords[1:]
    widths = right - left
    valid = widths > 1e-12
    if not np.any(valid):
        return np.zeros((0, 1), dtype=float)

    left = left[valid]
    right = right[valid]
    widths = widths[valid]
    slopes = np.abs(np.diff(unique_values)[valid] / widths)

    interval_weights = slopes.copy()
    if np.allclose(interval_weights, 0):
        interval_weights = widths.copy()
    if np.allclose(interval_weights, 0):
        interval_weights = np.ones_like(interval_weights)
    interval_weights = interval_weights / np.sum(interval_weights)

    lower_bound = float(discret_spl[0][0])
    upper_bound = float(discret_spl[0][-1])
    proposals = np.zeros((new_budget, 1), dtype=float)

    for i in range(new_budget):
        interval_idx = rng.choice(len(interval_weights), p=interval_weights)
        midpoint = 0.5 * (left[interval_idx] + right[interval_idx])
        sigma = max(widths[interval_idx] / 6.0, 1e-12)
        candidate = midpoint + rng.normal(scale=sigma)
        candidate = np.clip(candidate, left[interval_idx], right[interval_idx])
        candidate = np.clip(candidate, lower_bound, upper_bound)
        proposals[i, 0] = candidate

    return proposals


def _score_1d_gradient_candidates(samples_tri, values, candidate_points):
    if candidate_points.size == 0:
        return np.zeros(0, dtype=float)

    coords = samples_tri.reshape(-1).astype(float)
    values = np.asarray(values, dtype=float).reshape(-1)

    order = np.argsort(coords)
    coords = coords[order]
    values = values[order]

    unique_coords, inverse = np.unique(coords, return_inverse=True)
    unique_values = np.zeros(unique_coords.shape[0], dtype=float)
    counts = np.zeros(unique_coords.shape[0], dtype=float)
    for idx, group_idx in enumerate(inverse):
        unique_values[group_idx] += values[idx]
        counts[group_idx] += 1
    unique_values = unique_values / np.maximum(counts, 1.0)

    if unique_coords.shape[0] < 2:
        return np.zeros(candidate_points.shape[0], dtype=float)

    slopes = np.abs(np.diff(unique_values) / np.maximum(np.diff(unique_coords), 1e-12))
    interval_ids = np.searchsorted(unique_coords, candidate_points.reshape(-1), side="right") - 1
    interval_ids = np.clip(interval_ids, 0, slopes.shape[0] - 1)
    return slopes[interval_ids]


def _score_gradient_candidates(samples_tri, gradients, candidate_points, dim_weights=None, causal_mode=0):
    if candidate_points.size == 0:
        return np.zeros(0, dtype=float)

    if gradients.ndim == 1:
        return np.abs(gradients).astype(float)

    local_dim_weights = None if dim_weights is None else np.asarray(dim_weights, dtype=float).reshape(-1)
    apply_weighted_norm = local_dim_weights is not None and causal_mode in (0, 2)

    linear_interp = LinearNDInterpolator(samples_tri, gradients, fill_value=np.nan)
    grad_values = linear_interp(candidate_points)

    grad_values = np.asarray(grad_values, dtype=float)
    if grad_values.ndim == 1:
        grad_values = grad_values.reshape(-1, gradients.shape[1])

    missing = np.isnan(grad_values).any(axis=1)
    if np.any(missing):
        nearest_interp = NearestNDInterpolator(samples_tri, gradients)
        grad_values[missing] = nearest_interp(candidate_points[missing])

    if apply_weighted_norm:
        return np.sqrt(np.sum((grad_values * local_dim_weights) ** 2, axis=1))
    return np.linalg.norm(grad_values, axis=1)


def _execute_strategy_1(ndim_spl, discret_spl, total_budget, split, samples_output, rng, causal_weights=None, causal_mode=0):
    samples_tri = samples_output[:, np.array(split) - 1]
    weighted_means = samples_output[:, -1]

    unique_points, inverse = np.unique(samples_tri, axis=0, return_inverse=True)
    averaged_values = np.zeros(unique_points.shape[0])
    counts = np.zeros(unique_points.shape[0])
    for idx, group_idx in enumerate(inverse):
        averaged_values[group_idx] += weighted_means[idx]
        counts[group_idx] += 1
    averaged_values = averaged_values / np.maximum(counts, 1)
    samples_tri = unique_points

    local_dim_weights = None if causal_weights is None else np.asarray(causal_weights)[np.array(split) - 1]

    if ndim_spl == 1:
        proposed = _propose_1d_gradient_samples(samples_tri, averaged_values, discret_spl, total_budget, rng)
        scores = _score_1d_gradient_candidates(samples_tri, averaged_values, proposed)
        return proposed, scores

    min_required = ndim_spl + 1
    if samples_tri.shape[0] < min_required:
        return np.zeros((0, ndim_spl), dtype=float), np.zeros(0, dtype=float)

    try:
        tri = Delaunay(samples_tri)
        gradients = _estimate_gradient(tri, np.column_stack([samples_tri[:, :ndim_spl], averaged_values]))
        samples_tri_prop, _ = _propose_samples_new4(
            samples_tri=samples_tri,
            values=gradients,
            new_budget=total_budget,
            discret_spl=discret_spl,
            rng=rng,
            dim_weights=local_dim_weights,
            causal_mode=causal_mode,
        )
        scores = _score_gradient_candidates(
            samples_tri=samples_tri,
            gradients=gradients,
            candidate_points=samples_tri_prop,
            dim_weights=local_dim_weights,
            causal_mode=causal_mode,
        )
        return samples_tri_prop, scores
    except Exception as e:
        print(f"Skipping recommendations for split {split}: {e}")
        return np.zeros((0, ndim_spl), dtype=float), np.zeros(0, dtype=float)


def _merge_subspace_samples(split_samples, split_scores, splits, n_dims, taken_keys):
    if any(sample is None or sample.size == 0 for sample in split_samples):
        return np.zeros((0, n_dims), dtype=float), np.zeros(0, dtype=float), taken_keys

    merged_rows = []
    merged_scores = []
    split_entries = [list(zip(samples, scores)) for samples, scores in zip(split_samples, split_scores)]
    for row_group in itertools.product(*split_entries):
        x_full = np.zeros(n_dims, dtype=float)
        total_score = 0.0
        for split_idx, split in enumerate(splits):
            sample_values, score = row_group[split_idx]
            x_full[np.array(split) - 1] = sample_values
            total_score += float(score)

        key = "_".join([f"{value:.12g}" for value in x_full])
        if key in taken_keys:
            continue

        taken_keys.add(key)
        merged_rows.append(x_full)
        merged_scores.append(total_score)

    if not merged_rows:
        return np.zeros((0, n_dims), dtype=float), np.zeros(0, dtype=float), taken_keys
    return np.vstack(merged_rows), np.asarray(merged_scores, dtype=float), taken_keys


def _snap_to_domain(values, dim_name, dim_config, hp_dtypes):
    min_val = dim_config["min_val"]
    max_val = dim_config["max_val"]
    dtype = hp_dtypes.get(dim_name)

    clipped = np.clip(values, min_val, max_val)
    if dtype == "integer":
        return np.round(clipped).astype(int)
    return clipped.astype(float)


def run_g2s_causal_recommendation(sample_frame, dimensions, hp_dtypes, max_points, causal_mode=0, random_seed=RANDOM_SEED):
    """
    Generate new recommendations using the G2S sampler logic and the already observed samples.

    Args:
        sample_frame (pd.DataFrame): One column per recommended hyperparameter plus an ``outcome`` column.
        dimensions (dict): Maps hyperparameter name -> {strength, min_val, max_val}.
        hp_dtypes (dict): Maps hyperparameter name -> datatype.
        max_points (int): Maximum recommendation budget.
        causal_mode (int): 0 = weighted norm and anisotropic proposal, 1 = proposal only, 2 = norm only.
        random_seed (int): Random seed for reproducibility.

    Returns:
        list[tuple]: Recommended points with trailing estimated gradient score.
    """
    if not dimensions or max_points <= 0:
        return []

    if not isinstance(sample_frame, pd.DataFrame):
        raise TypeError("sample_frame must be a pandas DataFrame")

    dim_names = list(dimensions.keys())
    hp_columns = [f"HP.{name}" if f"HP.{name}" in sample_frame.columns else name for name in dim_names]
    required_columns = hp_columns + ["outcome"]
    missing_columns = [column for column in required_columns if column not in sample_frame.columns]
    if missing_columns:
        raise ValueError(f"sample_frame is missing required columns: {missing_columns}")

    working_df = sample_frame[required_columns].dropna().copy()
    if working_df.empty:
        return []

    rng = np.random.default_rng(random_seed)
    causal_weights = np.array([max(abs(dimensions[name]["strength"]), 1e-6) for name in dim_names], dtype=float)
    causal_weights = causal_weights / np.mean(causal_weights)

    discret = []
    for name in dim_names:
        dim_config = dimensions[name]
        discret.append(np.array([float(dim_config["min_val"]), float(dim_config["max_val"])], dtype=float))

    samples_output = working_df.to_numpy(dtype=float)
    existing_points = samples_output[:, :-1].copy()

    splits = _create_subspaces(len(dim_names))
    ndims_run, discrets_run, dims_left_run = _initialize_splits(splits, discret)

    taken = {"_".join([f"{value:.12g}" for value in row]) for row in existing_points}

    n_subspaces = max(len(splits), 1)
    subspace_budget = max(1, int(np.ceil(max_points ** (1.0 / n_subspaces))))

    proposed_by_split = [None for _ in splits]
    score_by_split = [None for _ in splits]
    for i in range(len(splits)):
        split, ndim_spl, discret_spl, _ = _get_split_information(
            splits,
            ndims_run,
            discrets_run,
            dims_left_run,
            i,
        )
        proposed_by_split[i], score_by_split[i] = _execute_strategy_1(
            ndim_spl=ndim_spl,
            discret_spl=discret_spl,
            total_budget=subspace_budget,
            split=split,
            samples_output=samples_output,
            rng=rng,
            causal_weights=causal_weights,
            causal_mode=causal_mode,
        )

    merged_points, merged_scores, _ = _merge_subspace_samples(
        proposed_by_split,
        score_by_split,
        splits,
        len(dim_names),
        taken,
    )
    if merged_points.size == 0:
        return []

    snapped_columns = []
    for idx, dim_name in enumerate(dim_names):
        snapped_columns.append(_snap_to_domain(merged_points[:, idx], dim_name, dimensions[dim_name], hp_dtypes))
    candidate_points = np.column_stack(snapped_columns)

    existing_df = pd.DataFrame(existing_points, columns=dim_names)
    candidate_df = pd.DataFrame(candidate_points, columns=dim_names).drop_duplicates().reset_index(drop=True)
    candidate_df = candidate_df.merge(existing_df.drop_duplicates(), on=dim_names, how="left", indicator=True)
    candidate_df = candidate_df[candidate_df["_merge"] == "left_only"].drop(columns=["_merge"]).reset_index(drop=True)
    if candidate_df.empty:
        return []

    score_df = pd.DataFrame(candidate_points, columns=dim_names)
    score_df["gradient_score"] = merged_scores
    score_df = score_df.groupby(dim_names, as_index=False)["gradient_score"].max()
    candidate_df = candidate_df.merge(score_df, on=dim_names, how="left")

    candidate_points = candidate_df.to_numpy(dtype=float)

    ranked = []
    for row in candidate_df.itertuples(index=False):
        values = []
        for dim_name in dim_names:
            point_value = getattr(row, dim_name)
            if hp_dtypes.get(dim_name) == "integer":
                values.append(int(round(point_value)))
            else:
                values.append(round(float(point_value), 8))
        ranked.append(tuple(values + [round(float(row.gradient_score), 8)]))

    ranked = sorted(ranked, key=lambda row: (-row[-1], row[:-1]))
    return ranked[:max_points]
