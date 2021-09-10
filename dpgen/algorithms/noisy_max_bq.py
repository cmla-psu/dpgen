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
    q, dq = all_inputs[:LENGTH], all_inputs[LENGTH:2 * LENGTH]
    noise = all_inputs[2 * LENGTH: 4 * LENGTH]
    return q, dq, noise


@numba.njit
def unpack_alignments(alignments: np.ndarray):
    proof, holes = alignments[:16], alignments[16:]

    return proof.astype(np.int8), holes.astype(np.int8)


def convert_alignment_str(alignments):
    proof, holes = unpack_alignments(alignments)
    eta_dist_true = sp.simplify(f"{proof[4]} + {proof[5]} * dq_i + {proof[6]} * bq_aligned_dist")
    eta_dist_false = sp.simplify(f"{proof[7]} + {proof[8]} * dq_i + {proof[9]} * bq_aligned_dist")
    eta = sp.simplify(f"{holes[0]} + {holes[1]} * LENGTH")
    eta2_dist_true = sp.simplify(f"{proof[10]} + {proof[11]} * dq_i + {proof[12]} * bq_aligned_dist")
    eta2_dist_false = sp.simplify(f"{proof[13]} + {proof[14]} * dq_i + {proof[15]} * bq_aligned_dist")
    eta2 = sp.simplify(f"{holes[2]} + {holes[3]} * LENGTH")
    return f"selector: Omega ? {proof[0] == 0} : {proof[1] == 0}, " \
           f"eta: Omega ? {eta_dist_true} : {eta_dist_false}, " \
           f"selector2: Omega ? {proof[2] == 0} : {proof[3] == 0} | " \
           f"eta2: Omega ? {eta2_dist_true} : {eta2_dist_false}", \
           f"eta: {eta}," \
           f"eta2: {eta2}"


@numba.njit(fastmath=True)
def noisy_max(all_inputs, alignments):
    # unpack the specific inputs from all_inputs
    q, dq, noise = unpack_inputs(all_inputs)
    proof, holes = unpack_alignments(alignments)

    # TODO: avoiding zero division problem
    if holes[0] + holes[1] * LENGTH < 1e-5 or holes[2] + holes[3] * LENGTH < 1e-5:
        return 1000 * EPSILON, 10, 0

    cost = 0
    failures = 0

    i, bq, imax = 0, 0, 0
    bq_aligned_dist, bq_shadow_dist, imax_aligned_dist, imax_shadow_dist = 0, 0, 0, 0

    while i < LENGTH:
        eta = noise[i] if np.abs(proof[4] + proof[5] * dq[i] + proof[6] * bq_aligned_dist) > 1e-3 or np.abs(
            proof[7] + proof[8] * dq[i] + proof[9] * bq_aligned_dist) > 1e-3 else 0
        eta2 = noise[2 * i] if np.abs(proof[8] + proof[9] * dq[i] + proof[10] * bq_aligned_dist) > 1e-3 or np.abs(
            proof[11] + proof[12] * dq[i] + proof[13] * bq_aligned_dist) > 1e-3 else 0
        eta_dist = proof[4] + proof[5] * dq[i] + proof[6] * bq_aligned_dist if q[i] + eta > bq + eta2 or i == 0 else \
        proof[7] + proof[8] * dq[i] + proof[9] * bq_aligned_dist
        eta2_dist = proof[10] + proof[11] * dq[i] + proof[12] * bq_aligned_dist if q[i] + eta > bq + eta2 or i == 0 else \
        proof[13] + proof[14] * dq[i] + proof[15] * bq_aligned_dist
        selector = proof[0] if q[i] + eta > bq + eta2 or i == 0 else proof[1]
        selector2 = proof[2] if q[i] + eta > bq + eta2 or i == 0 else proof[3]

        cost = (cost if selector == 0 or selector2 == 0 else 0) + np.abs(eta_dist) * (
                    EPSILON / (holes[0] + holes[1] * LENGTH)) + np.abs(eta2_dist) * (
                           EPSILON / (holes[2] + holes[3] * LENGTH))
        if selector == 1 or selector2 == 1:
            bq_aligned_dist = bq_shadow_dist
            imax_aligned_dist = imax_shadow_dist
        if q[i] + eta > bq + eta2 or i == 0:
            failures += my_assert(q[i] + dq[i] + eta + eta_dist > bq + eta2 + eta2_dist + bq_aligned_dist or i == 0)
            # failures += my_assert(np.abs(imax_aligned_dist) < 1e-5)
            imax_shadow_dist = imax + imax_shadow_dist - i
            imax = i
            imax_aligned_dist = 0
            bq_shadow_dist = bq + bq_shadow_dist - (q[i] + eta)
            bq = q[i]
            bq_aligned_dist = dq[i]
        else:
            failures += my_assert(
                not (q[i] + dq[i] + eta + eta_dist > bq + eta2 + eta2_dist + bq_aligned_dist or i == 0))

        # shadow execution
        if q[i] + dq[i] + eta > bq + bq_shadow_dist or i == 0:
            bq_shadow_dist = q[i] + dq[i] + eta - bq
            imax_shadow_dist = i - imax

        i = i + 1

    failures += my_assert(np.abs(imax_aligned_dist) < 1e-5)

    if cost - EPSILON > 1e-5:
        failures += 1

    return cost, failures, (np.square((holes[0] + holes[1] * LENGTH) / EPSILON) + np.square(
        (holes[2] + holes[3] * LENGTH) / EPSILON)) / (2 * np.square((9 + 9 * LENGTH) / EPSILON))


@numba.njit(parallel=True, fastmath=True)
def find_inputs(all_inputs, alignments):
    # bootstrap the process
    # for each particle
    results = np.empty(all_inputs.shape[0])
    for i in numba.prange(all_inputs.shape[0]):
        # TODO: this is the same as setting a constraint stating N <= LENGTH / 5
        dq = all_inputs[i][LENGTH:2 * LENGTH]
        if np.linalg.norm(dq, ord=1) < 0.5 * LENGTH:  # NOTICE: this constrains norm(dq) > 0.8 * LENGTH
            results[i] = 1000
        else:
            cost, failures, variance = noisy_max(all_inputs[i], alignments)
            results[i] = -failures

    return results


@numba.njit(parallel=True, fastmath=True)
def find_alignments(alignments, all_inputs: np.ndarray):
    # bootstrap the process
    # for each particle
    results = np.empty(alignments.shape[0])
    for i in numba.prange(alignments.shape[0]):
        cost, failures, variance = noisy_max(all_inputs, alignments[i])
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
        np.array([-10 for _ in range(LENGTH)] + [-1 for _ in range(LENGTH)] + [-10 for _ in range(2 * LENGTH)]))
    max_bounds = (np.array([10 for _ in range(LENGTH)] + [1 for _ in range(LENGTH)] + [10 for _ in range(2 * LENGTH)]))
    bounds = (min_bounds, max_bounds)

    align_min_bounds = [0, 0, 0, 0] + [-5 for _ in range(12)] + [0, 0, 0, 0]
    align_max_bounds = [2, 2, 2, 2] + [5 for _ in range(12)] + [5, 10, 5, 10]
    alignment_bounds = (align_min_bounds, align_max_bounds)
    # q, dq, noise
    alignment = np.array([0 for _ in range(16)] + [1, 1, 1, 1])
    # np.array([1, 0, 2, 0, 0, 0, 0, 0, 2, 0])
    print(convert_alignment_str(alignment))
    counterexamples = []  # [(np.array([0 for _ in range(LENGTH)] + [-1 for _ in range(LENGTH)] + [0 for _ in range(LENGTH + 1)] + [0, 15]))]
    iterations = 0
    while True:
        print(f'{iterations} | Searching for cex')
        optimizer = ps.single.GlobalBestPSO(n_particles=10000, dimensions=len(max_bounds), options=options,
                                            bounds=bounds, ftol=0.1, ftol_iter=20)
        cost, pos = optimizer.optimize(lambda x: find_inputs(x, alignment), iters=500)
        counterexamples.append(np.array(pos))
        print(noisy_max(pos, alignment))
        if cost > -1e-03:
            print(f'Final alignment: {convert_alignment_str(alignment)}')
            for cex in counterexamples:
                cost, failures, variance = noisy_max(cex, alignment)
                print(cost, failures)
            break
        iterations += 1
        print(f'{iterations} | Searching for alignment')
        # q, dq, noise = unpack_inputs(pos)
        # print(f'q+noise: {q + noise} | dq+noise: {q + dq + noise}')
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
