import math
import numpy as np
import itertools

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
    strengths = [dimensions[dim]['strength'] for dim in dimensions]
    product_strengths = math.prod(strengths)
    k = math.pow(max_points / product_strengths, 1.0 / len(strengths))

    for dim in dimensions:
        dimensions[dim]['point_count'] = max(1, math.floor(k * dimensions[dim]['strength']))

    total_points = math.prod([dimensions[dim]['point_count'] for dim in dimensions])

    return dimensions, total_points

def generate_grid_points(dimensions):
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

        if count == 1:
            grids.append([ (min_v + max_v) / 2.0 ])
        else:
            grids.append(np.linspace(min_v, max_v, count))

    grid_points = list(itertools.product(*grids))
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
    grid_points = generate_grid_points(dimensions)
    
    return grid_points, total_points