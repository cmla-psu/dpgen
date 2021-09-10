from typing import Callable

import numba
import numpy as np


def make_find_holes(f: Callable, inputs: np.ndarray):
    @numba.njit(parallel=True, fastmath=True)
    def find_holes(holes: np.ndarray):
        results = np.zeros(holes.shape[0])
        for cex in inputs:
            for i in numba.prange(holes.shape[0]):
                cost, failures, variance = f(cex, holes[i])
                # hard requirement
                if failures > 0:
                    results[i] = np.inf
                else:
                    # TODO: normalize this
                    results[i] += variance
            return results

    return find_holes


def make_find_inputs(f: Callable, holes: np.ndarray):
    @numba.njit(parallel=True, fastmath=True)
    def find_inputs(inputs: np.ndarray):
        results = np.empty(inputs.shape[0])
        for i in numba.prange(inputs.shape[0]):
            cost, failures, variance = f(inputs[i], holes)
            results[i] = -failures

        return results

    return find_inputs


"""
def search(program, query_length: int = 20):
    # swarm options
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    counterexamples = []
    iterations = 0
    input_optimizer = ps.single.GlobalBestPSO(n_particles=10000, dimensions=len(input_bounds_min), options=options, bounds=(input_bounds_min, input_bounds_max))
    holes_optimizer = ps.single.GlobalBestPSO(n_particles=10000, dimensions=len(holes_bounds_min), options=options, bounds=(holes_bounds_min, holes_bounds_max))

    holes = np.array([1 for _ in range(program.hole_count)])
    while True:
        print(f'{iterations} | Searching for cex')
        cost, pos = input_optimizer.optimize(make_find_inputs(program.wrap(query_length, noise_length), holes), iters=500)
        counterexamples.append(pos)
        if cost > -1e-03:
            #print(f'Final alignment: {convert_alignment_str(alignment)}')
            return holes
        iterations += 1
        print(f'{iterations} | Searching for alignment')
        #q, dq, noise, T, N = unpack_inputs(pos)
        #print(f'q+noise: {q + noise[1:]} | dq+noise: {q + dq + noise[1:]} | T+noise: {T + noise[0]} | N: {N}')

        cost, pos = holes_optimizer.optimize(make_find_holes(program.wrap(query_length, noise_length), np.asarray(counterexamples)), iters=500)
        holes = pos
        #proof, holes = unpack_alignments(pos)
        #alignment = np.concatenate((proof, holes))
        #cost_algo, failures, count_true, count_false, variance = svt(counterexamples[0], alignment)

        #print(cost_algo, failures, count_true, count_false, variance)
        #_, dq, *_ = unpack_inputs(counterexamples[0])
        #print(f'dq: {dq}')
        #print(alignment)
        #proof_str, hole_str = convert_alignment_str(alignment)
        #print(proof_str, hole_str)
        iterations += 1
"""
