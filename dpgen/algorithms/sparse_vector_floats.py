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
    noise, T, N = all_inputs[2 * LENGTH: 3 * LENGTH + 1], all_inputs[3 * LENGTH + 1], int(all_inputs[3 * LENGTH + 2])
    return q, dq, noise, T, N


@numba.njit
def unpack_alignments(alignments: np.ndarray):
    proof, holes = alignments[:7], alignments[7:]

    return proof.astype(np.int8), holes.astype(np.float32)


def convert_alignment_str(alignments):
    proof, holes = unpack_alignments(alignments)
    eta2_true_dist = sp.simplify(
        f"{proof[1]} + {proof[2] * proof[0]} + {proof[3]} * q_i_dist")

    eta2_false_dist = sp.simplify(
        f"{proof[4]} + {proof[5] * proof[0]} + {proof[6]} * q_i_dist")

    eta2 = sp.simplify(
        f"{holes[1]} + {holes[2]} * c"
    )

    return f"alignments: eta1: {proof[0]}, eta2: Omega ? {eta2_true_dist} : {eta2_false_dist}", f"eta1: {holes[0]} | eta2: {eta2}"


@numba.njit(fastmath=True)
def svt(all_inputs, alignments):
    # unpack the specific inputs from all_inputs
    q, dq, noise, T, c = unpack_inputs(all_inputs)
    proof, holes = unpack_alignments(alignments)

    # TODO: avoiding zero division problem
    if holes[0] < 1e-5 or holes[1] + holes[2] * c < 1e-5:
        return 1000 * EPSILON, 10, 0

    cost = 0
    failures = 0
    T_bar = T + noise[0]
    dist_T_bar = (proof[0])
    cost += np.abs(dist_T_bar) * (EPSILON / holes[0])

    i = 0
    count = 0
    while i < LENGTH and count < c:
        if q[i] + noise[i + 1] >= T_bar:
            eta2_dist = (proof[1] + proof[2] * dist_T_bar + proof[3] * dq[i])
            failures += my_assert(q[i] + dq[i] + noise[i + 1] + eta2_dist >= T_bar + dist_T_bar)
            cost += np.abs(eta2_dist) * EPSILON / (holes[1] + holes[2] * c)
            # output True
            count += 1
        else:
            eta2_dist = (proof[4] + proof[5] * dist_T_bar + proof[6] * dq[i])
            failures += my_assert(q[i] + dq[i] + noise[i + 1] + eta2_dist < T_bar + dist_T_bar)
            cost += np.abs(eta2_dist) * EPSILON / (holes[1] + holes[2] * c)
        i += 1

    if cost - EPSILON > 1e-5:
        failures += 1

    return cost, failures, (np.square(holes[0] / EPSILON) + np.square((holes[1] + holes[2] * c) / EPSILON)) / (
                np.square(9 / EPSILON) + np.square((9 + 9 * c) / EPSILON))


@numba.njit(parallel=True, fastmath=True)
def find_inputs(all_inputs, alignments):
    # bootstrap the process
    # for each particle
    results = np.empty(all_inputs.shape[0])
    for i in numba.prange(all_inputs.shape[0]):
        # TODO: this is the same as setting a constraint stating N <= LENGTH / 5
        N = int(all_inputs[i][3 * LENGTH + 2])
        dq = all_inputs[i][LENGTH:2 * LENGTH]
        # if N > int(LENGTH / 5)or np.linalg.norm(dq, ord=1) < 0.5 * LENGTH: # NOTICE: this constrains norm(dq) > 0.8 * LENGTH
        # results[i] = 1000 * N
        # else:
        # NOTICE: this does not set N < size /5
        cost, failures, variance = svt(all_inputs[i], alignments)
        results[i] = -failures

    return results


@numba.njit(parallel=True, fastmath=True)
def find_alignments(alignments, all_inputs: np.ndarray):
    # bootstrap the process
    # for each particle
    results = np.empty(alignments.shape[0])
    for i in numba.prange(alignments.shape[0]):
        cost, failures, variance = svt(all_inputs, alignments[i])
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


    """
    start = time.time()
    # Initialize swarm
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    min_bounds = (np.array(
        [-10 for _ in range(LENGTH)] + [-1 for _ in range(LENGTH)] + [-10 for _ in range(LENGTH + 1)] + [-10, 1]))
    max_bounds = (np.array(
        [10 for _ in range(LENGTH)] + [1 for _ in range(LENGTH)] + [10 for _ in range(LENGTH + 1)] + [10,
                                                                                                      2]))  # NOTICE: this effectively sets N == 1
    bounds = (min_bounds, max_bounds)

    align_min_bounds = [-5 for _ in range(7)] + [0, 0, 0]
    align_max_bounds = [5 for _ in range(7)] + [5, 10, 10]
    alignment_bounds = (align_min_bounds, align_max_bounds)
    # q, dq, noise, T, N
    alignment = np.array([0 for _ in range(7)] + [1, 1, 0])
    counterexamples = [
        (np.array([0 for _ in range(LENGTH)] + [-1 for _ in range(LENGTH)] + [0 for _ in range(LENGTH + 1)] + [0, 1]))]
    iterations = 0
    while True:
        print(f'{iterations} | Searching for cex')
        optimizer = ps.single.GlobalBestPSO(n_particles=10000, dimensions=3 * LENGTH + 3, options=options,
                                            bounds=bounds, ftol=0.1, ftol_iter=20)
        cost, pos = optimizer.optimize(lambda x: find_inputs(x, alignment), iters=500)
        counterexamples.append(np.array(pos))
        if cost > -1e-03:
            print(f'Final alignment: {convert_alignment_str(alignment)}')
            for cex in counterexamples:
                cost, failures, variance = svt(cex, alignment)
                print(cost, failures)
            break
        iterations += 1
        print(f'{iterations} | Searching for alignment')
        q, dq, noise, T, N = unpack_inputs(pos)
        print(f'q+noise: {q + noise[1:]} | dq+noise: {q + dq + noise[1:]} | T+noise: {T + noise[0]} | N: {N}')
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
