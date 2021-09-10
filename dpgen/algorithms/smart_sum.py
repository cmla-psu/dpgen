import time

import numba
import numpy as np
import pyswarms as ps
import sympy as sp

LENGTH = 100
EPSILON = 1


@numba.njit
def my_assert(cond):
    if not cond:
        return 1
    else:
        return 0


@numba.njit
def unpack_inputs(all_inputs):
    q, dq, index, M, T = all_inputs[:LENGTH], all_inputs[LENGTH:2 * LENGTH], int(all_inputs[2 * LENGTH]), int(
        all_inputs[2 * LENGTH + 1]), int(all_inputs[2 * LENGTH + 2])
    noise = all_inputs[2 * LENGTH + 3: 3 * LENGTH + 3]
    return q, dq, noise, index, M, T


@numba.njit
def unpack_alignments(alignments: np.ndarray):
    proof, holes = alignments[:8], alignments[8:]

    return proof.astype(np.int8), holes.astype(np.int8)


def convert_alignment_str(alignments):
    proof, holes = unpack_alignments(alignments)

    eta1 = sp.simplify(f'({holes[0]} + {holes[1]} * T + {holes[2]} * M + {holes[3]} * LENGTH)')
    eta1_dist = sp.simplify(f'{proof[0]} + {proof[1]} * vsum_dist + {proof[2]} * dq_i + {proof[3]} * vnext_dist')

    eta2 = sp.simplify(f'({holes[4]} + {holes[5]} * T + {holes[6]} * M + {holes[7]} * LENGTH)')
    eta2_dist = sp.simplify(f'{proof[4]} + {proof[5]} * vsum_dist + {proof[6]} * dq_i + {proof[7]} * vnext_dist')

    return f"alignments: eta1: {eta1_dist} | eta2: {eta2_dist}", f"eta1: {eta1} | eta2: {eta2}"


@numba.njit(fastmath=True)
def smartsum(all_inputs, alignments):
    # unpack the specific inputs from all_inputs
    q, dq, noise, index, M, T = unpack_inputs(all_inputs)
    proof, holes = unpack_alignments(alignments)

    # TODO: avoiding zero division problem
    if (holes[0] + holes[1] * T + holes[2] * M + holes[3] * LENGTH) < 1e-5 or (
            holes[4] + holes[5] * T + holes[6] * M + holes[7] * LENGTH) < 1e-5:
        return 1000 * EPSILON, 10, 0

    cost = 0
    failures = 0
    vnext, i, vsum = 0, 0, 0
    vsum_dist, vnext_dist = 0, 0
    while i < LENGTH and i <= T:
        converted_dq_i = dq[i] if index == i else 0
        if (i + 1) % M == 0:
            eta1 = noise[i]
            eta1_dist = proof[0] + proof[1] * vsum_dist + proof[2] * converted_dq_i + proof[3] * vnext_dist
            cost += np.abs(eta1_dist) * EPSILON / (holes[0] + holes[1] * T + holes[2] * M + holes[3] * LENGTH)
            vnext = vsum + q[i] + eta1
            vnext_dist = vsum_dist + converted_dq_i + eta1_dist
            vsum = 0
            vsum_dist = 0
            failures += my_assert(np.abs(vnext_dist) < 1e-5)
            # out = next::out
        else:
            eta2 = noise[i]
            eta2_dist = proof[4] + proof[5] * vsum_dist + proof[6] * converted_dq_i + proof[7] * vnext_dist
            cost += np.abs(eta2_dist) * EPSILON / (holes[4] + holes[5] * T + holes[6] * M + holes[7] * LENGTH)
            vnext += q[i] + eta2
            vnext_dist += converted_dq_i + eta2_dist
            vsum += q[i]
            vsum_dist += converted_dq_i
            failures += my_assert(np.abs(vnext_dist) < 1e-5)
            # out := next::out
        i += 1
    if cost - EPSILON > 1e-5:
        failures += 1

    return cost, failures, ((np.square(((holes[0] + holes[1] * T + holes[2] * M + holes[3] * LENGTH)) / EPSILON)) + (
        np.square((holes[4] + holes[5] * T + holes[6] * M + holes[7] * LENGTH) / EPSILON))) / (2 * (
        np.square((9 + 9 * T + 9 * M + 9 * LENGTH) / EPSILON)))  # / (np.square((9 + 9 * LENGTH) / EPSILON))


@numba.njit(parallel=True, fastmath=True)
def find_inputs(all_inputs, alignments):
    # bootstrap the process
    # for each particle
    results = np.empty(all_inputs.shape[0])
    for i in numba.prange(all_inputs.shape[0]):
        # dq = all_inputs[LENGTH:2 * LENGTH]
        # if np.linalg.norm(dq, ord=1) < 0.5 * LENGTH: # NOTICE: this constrains norm(dq) > 0.8 * LENGTH
        # results[i] = 1000
        # else:
        cost, failures, variance = smartsum(all_inputs[i], alignments)
        results[i] = -failures

    return results


@numba.njit(parallel=True, fastmath=True)
def find_alignments(alignments, all_inputs: np.ndarray):
    # bootstrap the process
    # for each particle
    results = np.empty(alignments.shape[0])
    for i in numba.prange(alignments.shape[0]):
        cost, failures, variance = smartsum(all_inputs, alignments[i])
        # hard requirement
        if failures > 0:
            results[i] = 100 * EPSILON * failures
        else:
            results[
                i] = variance  # + np.abs(cost - EPSILON) #- (count_false / LENGTH) + np.abs(count_true - N) / max(N, LENGTH - N)# - 20 * (count_true / LENGTH)
    return results


def main():
    """
    This *tries* to find the vanilla SVT, which is | eta1: 2 and eta2: 3c |, however, sometimes it finds | eta1: 5 and eta2: (2c + 2) |, which violates the privacy cost, so the LENGTH should be set higher
    also, we need to set N <= size / 5, moreover, we add a initial counterexample with dq = -1 and c = 15, to maximize the privacy cost, so that we can avoid the wrong solution.
    also, we set the bounds for eta1 to be (0, 5), so that we don't need to extend query length further
    with the above setting, we can reliably find ('alignments: eta1: 1, eta2: Omega ? 1 - q_i_dist : 0', 'eta1: 3 | eta2: 3*c')
    """
    start = time.time()
    # Initialize swarm
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    min_bounds = (
        np.array([-10 for _ in range(LENGTH)] + [-1 for _ in range(LENGTH)] + [0, 1, 0] + [-10 for _ in range(LENGTH)]))
    max_bounds = (np.array(
        [10 for _ in range(LENGTH)] + [1 for _ in range(LENGTH)] + [LENGTH, LENGTH, LENGTH] + [10 for _ in
                                                                                               range(LENGTH)]))
    bounds = (min_bounds, max_bounds)

    align_min_bounds = [-5 for _ in range(8)] + [0, 0, 0, 0, 0, 0, 0, 0]
    align_max_bounds = [5 for _ in range(8)] + [5, 10, 10, 10, 10, 10, 10, 10]
    alignment_bounds = (align_min_bounds, align_max_bounds)
    # q, dq, noise, T, N
    alignment = np.array([0 for _ in range(8)] + [1 for _ in range(8)])
    # alignment = np.array([0, -1, -1, 0, 0, 0, -1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    counterexamples = []  # [(np.array([0 for _ in range(LENGTH)] + [-1 for _ in range(LENGTH)] + [0 for _ in range(LENGTH + 1)] + [0, 15]))]
    iterations = 0
    while True:
        print(f'{iterations} | Searching for cex')
        optimizer = ps.single.GlobalBestPSO(n_particles=10000, dimensions=len(min_bounds), options=options,
                                            bounds=bounds, ftol=0.1, ftol_iter=20)
        cost, pos = optimizer.optimize(lambda x: find_inputs(x, alignment), iters=500)
        counterexamples.append(np.array(pos))
        if cost > -1e-03:
            print(f'Final alignment: {convert_alignment_str(alignment)}')
            for cex in counterexamples:
                cost, failures, variance = smartsum(cex, alignment)
                print(cost, failures)
            break
        iterations += 1
        print(f'{iterations} | Searching for alignment')
        # q, dq, noise, T, N = unpack_inputs(pos)
        # print(f'q+noise: {q + noise[1:]} | dq+noise: {q + dq + noise[1:]} | T+noise: {T + noise[0]} | N: {N}')
        optimizer = ps.single.GlobalBestPSO(n_particles=10000, dimensions=len(alignment), options=options,
                                            bounds=alignment_bounds)  # bounds=bounds)
        cost, pos = optimizer.optimize(lambda x: sum(find_alignments(x, all_inputs=cex) for cex in counterexamples),
                                       iters=500)
        proof, holes = unpack_alignments(pos)
        alignment = np.concatenate((proof, holes))
        proof_str, hole_str = convert_alignment_str(alignment)
        print(proof_str, hole_str)
        iterations += 1
    print(f'Total Time: {time.time() - start}s')
