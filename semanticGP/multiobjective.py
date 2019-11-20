"""Module containing tools that are useful when benchmarking algorithms
"""
from math import hypot, sqrt
from functools import wraps
from itertools import repeat
try:
    import numpy
except ImportError:
    numpy = False

try:
    # try importing the C version
    from ._hypervolume import hv
except ImportError:
    # fallback on python version
    from ._hypervolume import pyhv as hv
        
def diversity(first_front, first, last):
    """Given a Pareto front `first_front` and the two extreme points of the 
    optimal Pareto front, this function returns a metric of the diversity 
    of the front as explained in the original NSGA-II article by K. Deb.
    The smaller the value is, the better the front is.
    """
    df = hypot(first_front[0].fitness.values[0] - first[0],
               first_front[0].fitness.values[1] - first[1])
    dl = hypot(first_front[-1].fitness.values[0] - last[0],
               first_front[-1].fitness.values[1] - last[1])
    dt = [hypot(first.fitness.values[0] - second.fitness.values[0],
                first.fitness.values[1] - second.fitness.values[1])
          for first, second in zip(first_front[:-1], first_front[1:])]

    if len(first_front) == 1:
        return df + dl

    dm = sum(dt)/len(dt)
    di = sum(abs(d_i - dm) for d_i in dt)
    delta = (df + dl + di)/(df + dl + len(dt) * dm )
    return delta

def convergence(first_front, optimal_front):
    """Given a Pareto front `first_front` and the optimal Pareto front, 
    this function returns a metric of convergence
    of the front as explained in the original NSGA-II article by K. Deb.
    The smaller the value is, the closer the front is to the optimal one.
    """
    distances = []
    
    for ind in first_front:
        distances.append(float("inf"))
        for opt_ind in optimal_front:
            dist = 0.
            for i in range(len(opt_ind)):
                dist += (ind.fitness.values[i] - opt_ind[i])**2
            if dist < distances[-1]:
                distances[-1] = dist
        distances[-1] = sqrt(distances[-1])
        
    return sum(distances) / len(distances)


def hypervolume(front, ref=None):
    """Return the hypervolume of a *front*. If the *ref* point is not
    given, the worst value for each objective +1 is used.

    :param front: The population (usually a list of undominated individuals)
                  on which to compute the hypervolume.
    :param ref: A point of the same dimensionality as the individuals in *front*.
    """
    # Must use wvalues * -1 since hypervolume use implicit minimization
    wobj = numpy.array([ind.fitness.wvalues for ind in front]) * -1
    if ref is None:
        ref = numpy.max(wobj, axis=0) + 1
    return hv.hypervolume(wobj, ref)