import math
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler

from common.common_constants import RANDOM_SEED


# def is_likely_int(min_v, max_v):
#     return (
#         math.isclose(min_v, round(min_v), abs_tol=1e-9) and
#         math.isclose(max_v, round(max_v), abs_tol=1e-9) and
#         (max_v - min_v) >= 2
#     )


def distribute_points(dimensions, max_points):
    """
    Distributes points based on dimension configs.

    Args:
        dimensions (list of dict): Each dict has 'strength', 'min_val', 'max_val'.
        max_points (int): Max allowed total points.

    Returns:
        list of dict: dimensions with added 'point_count'.
        int: total points used.
    """
    strengths = {dim: abs(dimensions[dim]['strength']) for dim in dimensions}

    product_strengths = math.prod(strengths.values())
    k = math.pow(max_points / product_strengths, 1.0 / len(strengths))
    for dim in dimensions:
        dimensions[dim]['point_count'] = max(1, math.floor(k * strengths[dim]))
    
    return dimensions


def generate_grid_points(dimensions, hp_dtypes):
    """
    Generates grid points based on dimension configs with point counts.

    Args:
        dimensions (list of dict): Each dict has 'min_val', 'max_val', 'point_count'.

    Returns:
        list of tuples: grid points.
    """
    grids = []
    for dim in dimensions:
        min_v = dimensions[dim]['min_val']
        max_v = dimensions[dim]['max_val']
        count = dimensions[dim]['point_count']

        is_int = hp_dtypes[dim] == 'integer'

        if count == 1:
            mid = (min_v + max_v) / 2.0
            grids.append([int(round(mid))] if is_int else [mid])
        else:
            vals = np.linspace(min_v, max_v, count)
            grids.append(np.round(vals).astype(int).tolist() if is_int else vals.tolist())

    grid_points = list(itertools.product(*grids))

    return grid_points


def weight_recommendations(data, grid_points):
    # data and grid_points should be lists/arrays of shape (n_samples, n_features)
    scaler = StandardScaler()
    all_points = np.vstack([data, grid_points])
    scaler.fit(all_points)  # fit on combined set or only on data depending on choice

    data_s = scaler.transform(data)
    grid_s = scaler.transform(grid_points)

    grid_points_dist = []
    for gp, gp_s in zip(grid_points, grid_s):
        # compute distance to all current points in scaled space
        min_dist = round(float(np.linalg.norm(data_s - gp_s, axis=1).min()), 8)
        grid_points_dist.append(gp + (min_dist,))

    return grid_points_dist


def run_causal_recommendation(data, dimensions, hp_dtypes, max_points):
    """
    Main function to run causal recommendation.

    Args:
        dimensions (list of dict): Each dict has 'strength', 'min_val', 'max_val'.
        hp_dtypes: Hyperparameter datatype dictionary.
        max_points (int): Max allowed total points.

    Returns:
        list of tuples: grid points.
        int: total points used.
    """
    dimensions = distribute_points(dimensions, max_points)

    grid_points = generate_grid_points(dimensions, hp_dtypes)

    grid_points = weight_recommendations(data, grid_points)

    grid_points = sorted(grid_points, key=lambda x: (-x[-1], x[:-1]))
    
    return grid_points
