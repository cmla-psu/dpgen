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
    noise, T, N = all_inputs[2 * LENGTH: 4 * LENGTH + 1], all_inputs[4 * LENGTH + 1], int(all_inputs[4 * LENGTH + 2])
    return q, dq, noise, T, N


@numba.njit
def unpack_alignments(alignments: np.ndarray):
    proof, holes = alignments[:17], alignments[17:]

    return proof.astype(np.int8), holes.astype(np.int8)


def convert_alignment_str(alignments):
    proof, holes = unpack_alignments(alignments)
    eta2_true_dist = sp.simplify(
        f"{proof[1]} + {proof[2] * proof[0]} + {proof[3]} * q_i_dist")

    eta2_false_dist = sp.simplify(
        f"{proof[4]} + {proof[5] * proof[0]} + {proof[6]} * q_i_dist")

    eta2 = sp.simplify(
        f"{holes[1]} + {holes[2]} * c"
    )

    eta3_true_dist = sp.simplify(
        f"{proof[7]} + {proof[8] * proof[0]} + {proof[9]} * q_i_dist")

    eta3_false_dist = sp.simplify(
        f"{proof[10]} + {proof[11] * proof[0]} + {proof[12]} * q_i_dist")

    eta3 = sp.simplify(
        f"{holes[3]} + {holes[4]} * c"
    )

    while_cond = sp.simplify(
        f"Max(Abs({proof[13]}) * EPSILON / ({eta2}), Abs({proof[14]}) * EPSILON / ({eta2}), Abs({proof[15]}) * EPSILON / ({eta3}), Abs({proof[16]}) * EPSILON / ({eta3}))"
    )

    return f"alignments: eta1: {proof[0]}, eta2: Omega_top ? {eta2_true_dist} : {eta2_false_dist}, eta3: Omega_middle ? {eta3_true_dist} : {eta3_false_dist}", f"eta1: {holes[0]} | eta2: {eta2} | eta3: {eta3} | while: {while_cond}"


@numba.njit(fastmath=True)
def adaptivesvt_original(all_inputs, alignments):
    # unpack the specific inputs from all_inputs
    q, dq, noise, T, c = unpack_inputs(all_inputs)
    proof, holes = unpack_alignments(alignments)

    cost = 0
    failures = 0
    T_bar = T + np.random.laplace(0, (holes[0] / EPSILON)) if proof[0] != 0 else 0
    dist_T_bar = (proof[0])
    cost += np.abs(dist_T_bar) * (EPSILON / holes[0])
    i = 0
    sigma = 10
    true_positives, false_positives = 0, 0
    # [1, 1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 2, 0, 8, 0, 4]
    while i < LENGTH and cost <= EPSILON - max(
            np.abs(proof[13]) * EPSILON / (holes[1] + holes[2] * c),
            np.abs(proof[14]) * EPSILON / (holes[1] + holes[2] * c),
            np.abs(proof[15]) * EPSILON / (holes[3] + holes[4] * c),
            np.abs(proof[16]) * EPSILON / (holes[3] + holes[4] * c)
    ) + 1e-3:
        eta2 = np.random.laplace(0, (holes[1] + holes[2] * c) / EPSILON)
        if q[i] + eta2 - T_bar >= sigma:  # NOTICE: should be sigma
            cost += np.abs(proof[13]) * EPSILON / (holes[1] + holes[2] * c)
            if q[i] >= T:
                true_positives += 1
            else:
                false_positives += 1
        else:
            cost += np.abs(proof[14]) * EPSILON / (holes[1] + holes[2] * c)
            eta3 = np.random.laplace(0, (holes[3] + holes[4] * c) / EPSILON)
            if q[i] + eta3 - T_bar >= 0:
                if q[i] >= T:
                    true_positives += 1
                else:
                    false_positives += 1
                cost += np.abs(proof[15]) * EPSILON / (holes[3] + holes[4] * c)
            else:
                cost += np.abs(proof[16]) * EPSILON / (holes[3] + holes[4] * c)
        i += 1
    return true_positives, false_positives


@numba.njit(fastmath=True)
def adaptivesvt(all_inputs, alignments):
    # unpack the specific inputs from all_inputs
    q, dq, noise, T, c = unpack_inputs(all_inputs)
    proof, holes = unpack_alignments(alignments)

    # TODO: avoiding zero division problem
    if holes[0] < 1e-5 or holes[1] + holes[2] * c < 1e-5 or holes[3] + holes[4] * c < 1e-5:
        return 1000 * EPSILON, 10000, 0

    cost = 0
    failures = 0
    idx = 0
    T_bar = T + noise[idx]
    idx += 1
    dist_T_bar = (proof[0])
    cost += np.abs(dist_T_bar) * (EPSILON / holes[0])
    i = 0
    sigma = 5
    # top_queries = 0
    # [1, 1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 2, 0, 2, 0, 2, 0, 8, 0, 4]
    # [1, 1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 2, 0, 8, 0, 4]
    # [0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 1, 0, 3, 6, 5, 5, 4]
    # [1, 1, 0, -1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0]
    while i < LENGTH and cost <= EPSILON - max(
            np.abs(proof[13]) * EPSILON / (holes[1] + holes[2] * c),
            np.abs(proof[14]) * EPSILON / (holes[1] + holes[2] * c),
            np.abs(proof[15]) * EPSILON / (holes[3] + holes[4] * c),
            np.abs(proof[16]) * EPSILON / (holes[3] + holes[4] * c)
    ) + 1e-3:
        eta2 = noise[idx]
        idx += 1
        if q[i] + eta2 - T_bar >= sigma:  # NOTICE: should be sigma
            eta2_dist = (proof[1] + proof[2] * dist_T_bar + proof[3] * dq[i])
            if np.abs(proof[13]) < np.abs(eta2_dist):
                return 1000 * EPSILON, 10000, 0
            failures += my_assert(q[i] + dq[i] + eta2 + eta2_dist - (T_bar + dist_T_bar) >= sigma)

            cost += np.abs(proof[13]) * EPSILON / (holes[1] + holes[2] * c)
            failures += my_assert(np.abs(dq[i] + eta2_dist - dist_T_bar) < 1e-5)
            # top_queries += 1
        else:
            eta2_dist = (proof[4] + proof[5] * dist_T_bar + proof[6] * dq[i])
            if np.abs(proof[14]) < np.abs(eta2_dist):
                return 1000 * EPSILON, 10000, 0
            failures += my_assert(q[i] + dq[i] + eta2 + eta2_dist - (T_bar + dist_T_bar) < sigma)
            cost += np.abs(proof[14]) * EPSILON / (holes[1] + holes[2] * c)
            eta3 = noise[idx]
            idx += 1
            if q[i] + eta3 - T_bar >= 0:
                eta3_dist = (proof[7] + proof[8] * dist_T_bar + proof[9] * dq[i])
                if np.abs(proof[15]) < np.abs(eta3_dist):
                    return 1000 * EPSILON, 10000, 0
                failures += my_assert(q[i] + dq[i] + eta3 + eta3_dist - (T_bar + dist_T_bar) >= 0)
                cost += np.abs(proof[15]) * EPSILON / (holes[3] + holes[4] * c)
                failures += my_assert(np.abs(dq[i] + eta3_dist - dist_T_bar) < 1e-5)
                # top_queries += 1
            else:
                eta3_dist = (proof[10] + proof[11] * dist_T_bar + proof[12] * dq[i])
                if np.abs(proof[16]) < np.abs(eta3_dist):
                    return 1000 * EPSILON, 10000, 0
                failures += my_assert(q[i] + dq[i] + eta3 + eta3_dist - (T_bar + dist_T_bar) < 0)
                cost += np.abs(proof[16]) * EPSILON / (holes[3] + holes[4] * c)
        i += 1

    if i == 0:
        failures += 1

    if cost - EPSILON > 1e-3:
        failures += 1

    return cost, failures, (np.square(holes[0] / EPSILON) + np.square((holes[1] + holes[2] * c) / EPSILON) + np.square(
        (holes[3] + holes[4] * c) / EPSILON)) / (np.square(9 / EPSILON) + 2 * np.square((9 + 9 * c) / EPSILON))


@numba.njit(parallel=True, fastmath=True)
def find_inputs(all_inputs, alignments):
    # bootstrap the process
    # for each particle
    results = np.empty(all_inputs.shape[0])
    for i in numba.prange(all_inputs.shape[0]):
        # TODO: this is the same as setting a constraint stating N <= LENGTH / 5
        N = int(all_inputs[i][4 * LENGTH + 2])
        dq = all_inputs[i][LENGTH:2 * LENGTH]
        if N > int(LENGTH / 5) or np.linalg.norm(dq,
                                                 ord=1) < 0.5 * LENGTH:  # NOTICE: this constrains norm(dq) > 0.8 * LENGTH
            results[i] = 1e12 * N
        else:
            cost, failures, variance = adaptivesvt(all_inputs[i], alignments)
            results[i] = -failures

    return results


@numba.njit(parallel=True, fastmath=True)
def find_alignments(alignments: np.ndarray, all_inputs: np.ndarray):
    # bootstrap the process
    # for each particle
    c = int(all_inputs[0][4 * LENGTH + 2])

    results = np.zeros(alignments.shape[0])
    # alignments[0] = np.array([1, 1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 2, 0, 2, 0, 2, 0, 8, 0, 4])
    for i in numba.prange(alignments.shape[0]):
        for cex in all_inputs:
            cost, failures, variance = adaptivesvt(cex, alignments[i])
            # hard requirement
            if failures > 0:
                results[i] = 1e8 * EPSILON * failures
                break

        # passed all privacy checks, we run accuracy check
        if np.abs(results[i]) < 1e-3:
            # run the original svt multiple times to get estimates of #true positive and #false positive
            total = 5000
            true_positive, false_positive, penalty = 0, 0, 0
            for _ in range(total):
                local_true, local_false = adaptivesvt_original(all_inputs[0], alignments[i])
                true_positive += local_true
                false_positive += local_false
                # if local_true + local_false < 0.8 * c:
                # results[i] += 1e8 * (c - (local_true + local_false))
                # break

            results[i] += -true_positive / total + false_positive / total
        # results[i] = variance #+ np.abs(cost - EPSILON) #- (count_false / LENGTH) + np.abs(count_true - N) / max(N, LENGTH - N)# - 20 * (count_true / LENGTH)
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
    min_bounds = (np.array(
        [-10 for _ in range(LENGTH)] + [-1 for _ in range(LENGTH)] + [-100 for _ in range(2 * LENGTH + 1)] + [-10, 1]))
    max_bounds = (np.array(
        [10 for _ in range(LENGTH)] + [1 for _ in range(LENGTH)] + [100 for _ in range(2 * LENGTH + 1)] + [10, LENGTH]))
    bounds = (min_bounds, max_bounds)

    align_min_bounds = [-2 for _ in range(17)] + [0, 0, 0, 0, 0]
    align_max_bounds = [3 for _ in range(17)] + [5, 10, 10, 10, 10]
    alignment_bounds = (align_min_bounds, align_max_bounds)
    # q, dq, noise, T, N
    alignment = np.array([0 for _ in range(17)] + [1, 1, 0, 1, 0])
    # alignment = np.array([0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 1, 0, 3, 6, 5, 5, 4])
    # print(convert_alignment_str(alignment))
    # alignment = np.array([1, 1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 2, 0, 2, 0, 2, 0, 8, 0, 4])
    # alignment = np.array([1, 1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 2, 0, 2, 0, 4, 9, 9, 7, 9])
    # alignment = np.array([1, 0, 1, -1, 0, 0, 0, 0, 1, -1, 0, 0, 0, 2, 0, 2, 0, 3, 3, 9, 5, 9])
    # alignment = np.array([1, 1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 2, 0, 2, 0, 3, 0, 6, 0, 3])
    # alignment = np.array([1, 0, 1, -1, 0, 0, 0, 1, 0, -1, -1, 1, 0, 2, 0, 2, 0, 3, 7, 4, 6, 3])
    # counterexamples = [(np.array([0 for _ in range(LENGTH)] + [-1 for _ in range(LENGTH)] + [0 for _ in range(2 * LENGTH + 1)] + [0, 15]))]
    q = np.array([-1000 for _ in range(int(0.75 * LENGTH))] + [1000 for _ in range(int(0.1 * LENGTH))] + [50 for _ in
                                                                                                          range(
                                                                                                              int(0.15 * LENGTH))])
    counterexamples = [
        np.concatenate((q, np.array([0 for _ in range(LENGTH)] + [0 for _ in range(2 * LENGTH + 1)] + [0, 20]))).astype(
            float)]
    iterations = 0
    tp, fp = adaptivesvt_original(counterexamples[0], alignment)
    print(tp, fp)

    alignment_options = {'c1': 0.5, 'c2': 0.3, 'w': 2}
    oh_strategy = {"w": 'exp_decay', "c1": 'nonlin_mod', "c2": 'lin_variation'}
    while True:
        print(f'{iterations} | Searching for cex')
        optimizer = ps.single.GlobalBestPSO(n_particles=10000, dimensions=4 * LENGTH + 3, options=options,
                                            bounds=bounds, ftol=0.1, ftol_iter=50)
        cost, pos = optimizer.optimize(lambda x: find_inputs(x, alignment), iters=500)
        counterexamples.append(np.array(pos))
        if cost > -1e-03:
            print(f'Final Alignment: {alignment}')
            print(f'Final alignment: {convert_alignment_str(alignment)}')
            for cex in counterexamples:
                cost, failures, variance = adaptivesvt(cex, alignment)
                print(cost, failures)
            break
        iterations += 1
        print(f'{iterations} | Searching for alignment')
        # q, dq, noise, T, N = unpack_inputs(pos)
        # print(f'q+noise: {q + noise[1:]} | dq+noise: {q + dq + noise[1:]} | T+noise: {T + noise[0]} | N: {N}')
        optimizer = ps.single.GlobalBestPSO(n_particles=50000, dimensions=len(alignment), options=options,
                                            oh_strategy=oh_strategy, bounds=alignment_bounds, ftol=0.1, ftol_iter=30)
        cost, pos = optimizer.optimize(lambda x: find_alignments(x, all_inputs=np.asarray(counterexamples)), iters=500)
        proof, holes = unpack_alignments(pos)
        alignment = np.concatenate((proof, holes))
        proof_str, hole_str = convert_alignment_str(alignment)
        print(proof_str, hole_str)
        iterations += 1
    print(f'Total Time: {time.time() - start}s')
