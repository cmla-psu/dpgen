import numba
import numpy as np
import sympy as sp
from dpgen.transform.utils import dpgen_assert

from dpgen.core import search

dpgen_length = 20


@numba.njit
def sparse_vector_constraint(epsilon, q, t, c, dpgen_q_distance, dpgen_noise):
    return c <= len(q) / 5


@numba.njit
def sparse_vector(inputs, holes):
    epsilon = 1
    q = inputs[1:1 + dpgen_length]
    t, c = inputs[1 + dpgen_length], int(inputs[2 + dpgen_length])
    dpgen_q_distance, dpgen_noise = inputs[3 + dpgen_length:3 + 2 * dpgen_length], inputs[
                                                                                   3 + 2 * dpgen_length: 4 + 3 * dpgen_length]

    dpgen_proof, dpgen_holes = holes[:7].astype(np.int8), holes[7:].astype(np.int8)
    dpgen_cost, dpgen_idx, dpgen_failures = 0, 0, 0

    # custom constraint
    if not sparse_vector_constraint(epsilon, q, t, c, dpgen_q_distance, dpgen_noise):
        return 1000 * epsilon, -1000, 0

    if dpgen_holes[0] < 1e-3 or dpgen_holes[1] + dpgen_holes[2] * c < 1e-3:
        return 1000 * epsilon, 1000, 0

    dpgen_eta1 = dpgen_noise[dpgen_idx]
    dpgen_idx += 1
    dpgen_eta1_distance = (dpgen_proof[0])
    dpgen_cost += np.abs(dpgen_eta1_distance) * (epsilon / dpgen_holes[0])

    t_bar = t + dpgen_eta1
    dpgen_t_bar_distance = dpgen_eta1_distance

    i = 0
    count = 0
    while i < q.shape[0] and count < c:
        dpgen_eta2 = dpgen_noise[dpgen_idx]
        dpgen_idx += 1
        dpgen_eta2_distance = (
            dpgen_proof[1] + dpgen_proof[2] * dpgen_t_bar_distance + dpgen_proof[3] * dpgen_q_distance[i]
            if q[i] + dpgen_eta2 >= t_bar
            else (dpgen_proof[4] + dpgen_proof[5] * dpgen_t_bar_distance + dpgen_proof[6] * dpgen_q_distance[i])
        )
        dpgen_cost += np.abs(dpgen_eta2_distance) * epsilon / (dpgen_holes[1] + dpgen_holes[2] * c)
        if q[i] + dpgen_eta2 >= t_bar:
            dpgen_failures += dpgen_assert(
                q[i] + dpgen_q_distance[i] + dpgen_eta2 + dpgen_eta2_distance >= t_bar + dpgen_t_bar_distance
            )
            # output True
            count += 1
        else:
            dpgen_failures += dpgen_assert(
                q[i] + dpgen_q_distance[i] + dpgen_eta2 + dpgen_eta2_distance < t_bar + dpgen_t_bar_distance
            )
            # output False
        i += 1

    if dpgen_cost > epsilon:
        dpgen_failures += 1

    # (cost, failures, total variance)
    return dpgen_cost, dpgen_failures, np.square(dpgen_holes[0]) + np.square((dpgen_holes[1] + dpgen_holes[2] * c))


def convert_holes_str(holes):
    proof, holes = holes[:7].astype(np.int8), holes[7:].astype(np.int8)
    eta2_true_dist = sp.simplify(
        f"{proof[1]} + {proof[2] * proof[0]} + {proof[3]} * q_i_dist")

    eta2_false_dist = sp.simplify(
        f"{proof[4]} + {proof[5] * proof[0]} + {proof[6]} * q_i_dist")

    eta2 = sp.simplify(
        f"{holes[1]} + {holes[2]} * c"
    )

    return f"alignments: eta1: {proof[0]}, eta2: Omega ? {eta2_true_dist} : {eta2_false_dist}", f"eta1: {holes[0]} | eta2: {eta2}"


def main():
    min_bounds = (np.array(
        [0.99] + [-10 for _ in range(dpgen_length)] + [-10, 1] + [-1 for _ in range(dpgen_length)] + [-10 for _ in
                                                                                                      range(
                                                                                                          dpgen_length + 1)]))
    max_bounds = (np.array(
        [1] + [10 for _ in range(dpgen_length)] + [10, dpgen_length] + [1 for _ in range(dpgen_length)] + [10 for _ in
                                                                                                           range(
                                                                                                               dpgen_length + 1)]))
    bounds = (min_bounds, max_bounds)
    align_min_bounds = [-5 for _ in range(7)] + [0, 0, 0]
    align_max_bounds = [5 for _ in range(7)] + [10, 10, 10]
    alignment_bounds = (align_min_bounds, align_max_bounds)
    print(search(sparse_vector, bounds, alignment_bounds, convert_holes_str))
