import math
import random
import numpy as np
import itertools


def is_likely_int(min_v, max_v):
    return (
        math.isclose(min_v, round(min_v), abs_tol=1e-9) and
        math.isclose(max_v, round(max_v), abs_tol=1e-9) and
        (max_v - min_v) >= 2
    )


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

    total_points = math.prod([dimensions[dim]['point_count'] for dim in dimensions])

    return dimensions, total_points


def generate_grid_points(dimensions, max_points):
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

        is_int = is_likely_int(min_v, max_v)

        if count == 1:
            mid = (min_v + max_v) / 2.0
            grids.append([int(round(mid))] if is_int else [mid])
        else:
            vals = np.linspace(min_v, max_v, count)
            grids.append(np.round(vals).astype(int).tolist() if is_int else vals.tolist())

    grid_points = list(itertools.product(*grids))
    
    if len(grid_points) > max_points:
        rng = random.Random(42)
        grid_points = rng.sample(grid_points, max_points)
    
    grid_points = sorted(grid_points)

    return grid_points

def run_causal_recommendation(dimensions, max_points):
    """
    Main function to run causal recommendation.

    Args:
        dimensions (list of dict): Each dict has 'strength', 'min_val', 'max_val'.
        max_points (int): Max allowed total points.

    Returns:
        list of tuples: grid points.
        int: total points used.
    """
    dimensions, total_points = distribute_points(dimensions, max_points)
    grid_points = generate_grid_points(dimensions, max_points)
    
    return grid_points, total_points
