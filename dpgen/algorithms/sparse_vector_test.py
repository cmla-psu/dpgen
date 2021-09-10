import time

import numba
import numpy as np
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
    proof, holes = alignments[:9], alignments[9:]

    return proof.astype(np.int8), holes.astype(np.int8)


def convert_alignment_str(alignments):
    proof, holes = unpack_alignments(alignments)
    eta2_true_dist = sp.simplify(
        f"{proof[1]} + {proof[2] * proof[0]} + {proof[3]} * q_i_dist"
    )

    eta2_false_dist = sp.simplify(
        f"{proof[4]} + {proof[5] * proof[0]} + {proof[6]} * q_i_dist"
    )

    eta2 = sp.simplify(
        f"{holes[1]} + {holes[2]} * c"
    )

    while_cond = sp.simplify(
        f"Max(Abs({proof[7]}), Abs({proof[8]}))"
    )

    return f"alignments: eta1: {proof[0]}, eta2: Omega ? {eta2_true_dist} : {eta2_false_dist}", f"eta1: {holes[0]} | eta2: {eta2} | while: {while_cond} * ({eta2})"


@numba.njit(fastmath=True)
def svt_original(all_inputs, alignments):
    # unpack the specific inputs from all_inputs
    q, dq, noise, T, c = unpack_inputs(all_inputs)
    proof, holes = unpack_alignments(alignments)

    cost = 0
    T_bar = T + (np.random.laplace(0, holes[0] / EPSILON) if proof[0] != 0 else 0)
    dist_T_bar = (proof[0])
    cost += np.abs(dist_T_bar) * (EPSILON / holes[0])
    # [0, 1, 0, 0, 0, 0, -1, 1, 0, 3, 1, 5]
    i = 0
    true_positive, false_positive = 0, 0
    while i < q.shape[0] and cost <= EPSILON - max(np.abs((proof[7])) * EPSILON / (holes[1] + holes[2] * c),
                                                   np.abs((proof[8])) * EPSILON / (holes[1] + holes[2] * c)):
        eta2 = np.random.laplace(0, (holes[1] + holes[2] * c) / EPSILON)

        if q[i] + eta2 >= T_bar:
            eta2_dist = (proof[1] + proof[2] * dist_T_bar + proof[3] * dq[i])
            cost += np.abs(eta2_dist) * EPSILON / (holes[1] + holes[2] * c)
            if q[i] >= T:
                true_positive += 1
            else:
                false_positive += 1
        else:
            eta2_dist = (proof[4] + proof[5] * dist_T_bar + proof[6] * dq[i])
            cost += np.abs(eta2_dist) * EPSILON / (holes[1] + holes[2] * c)
        i += 1

    return true_positive, false_positive, i


@numba.njit(fastmath=True)
def svt(all_inputs, alignments):
    # unpack the specific inputs from all_inputs
    q, dq, noise, T, c = unpack_inputs(all_inputs)
    proof, holes = unpack_alignments(alignments)
    # TODO: avoiding zero division problem
    if holes[0] < 1e-5 or holes[1] + holes[2] * c < 1e-5:
        return 1000 * EPSILON, 10, 0, 0, 0, 0, 0

    cost = 0
    failures = 0
    T_bar = T + noise[0]
    dist_T_bar = (proof[0])
    cost += np.abs(dist_T_bar) * (EPSILON / holes[0])

    i = 0
    true_positive, false_positive = 0, 0
    true_negative, false_negative = 0, 0
    # count = 0
    # []
    while i < LENGTH and cost <= EPSILON - max(np.abs((proof[7])) * EPSILON / (holes[1] + holes[2] * c),
                                               np.abs((proof[8])) * EPSILON / (holes[1] + holes[2] * c)):
        if q[i] + noise[i + 1] >= T_bar:
            eta2_dist = (proof[1] + proof[2] * dist_T_bar + proof[3] * dq[i])
            if np.abs(proof[7]) < np.abs(eta2_dist):
                return 1000 * EPSILON, 10, 0, 0, 0, 0, 0
            failures += my_assert(q[i] + dq[i] + noise[i + 1] + eta2_dist >= T_bar + dist_T_bar)
            cost += np.abs(eta2_dist) * EPSILON / (holes[1] + holes[2] * c)
            false_positive += 1 if q[i] < T else 0
            true_positive += 1 if q[i] >= T else 0
            # output True
            # count += 1
        else:
            eta2_dist = (proof[4] + proof[5] * dist_T_bar + proof[6] * dq[i])
            if np.abs(proof[8]) < np.abs(eta2_dist):
                return 1000 * EPSILON, 10, 0, 0, 0, 0, 0
            failures += my_assert(q[i] + dq[i] + noise[i + 1] + eta2_dist < T_bar + dist_T_bar)
            cost += np.abs(eta2_dist) * EPSILON / (holes[1] + holes[2] * c)
            false_negative += 1 if q[i] >= T else 0
        i += 1

    # all_checks = q >= T
    # all_checks[:i] = False
    # false_negative += np.sum(all_checks)

    if cost - EPSILON > 1e-5:
        failures += 1

    return cost, failures, (np.square(holes[0] / EPSILON) + np.square((holes[1] + holes[2] * c) / EPSILON)) / (
                np.square(9 / EPSILON) + np.square(
            (9 + 9 * c) / EPSILON)), true_positive, false_positive, true_negative, false_negative


@numba.njit(parallel=True, fastmath=True)
def find_inputs(all_inputs, alignments):
    # bootstrap the process
    # for each particle
    results = np.empty(all_inputs.shape[0])
    for i in numba.prange(all_inputs.shape[0]):
        # TODO: this is the same as setting a constraint stating N <= LENGTH / 5
        N = int(all_inputs[i][3 * LENGTH + 2])
        dq = all_inputs[i][LENGTH:2 * LENGTH]
        if N > int(LENGTH / 5) or np.linalg.norm(dq,
                                                 ord=1) < 0.5 * LENGTH:  # NOTICE: this constrains norm(dq) > 0.8 * LENGTH
            results[i] = 1000 * N
        else:
            cost, failures, variance, true_positive, false_positive, true_negative, false_negative = svt(all_inputs[i],
                                                                                                         alignments)
        results[i] = -failures

    return results


@numba.njit(parallel=True, fastmath=True)
def find_alignments(alignments, all_inputs: np.ndarray):
    # bootstrap the process
    # for each particle
    results = np.empty(alignments.shape[0])
    c = int(all_inputs[3 * LENGTH + 2])
    for i in numba.prange(alignments.shape[0]):
        cost, failures, variance, true_positive, false_positive, true_negative, false_negative = svt(all_inputs,
                                                                                                     alignments[i])
        # hard requirement
        if failures > 0:
            results[i] = 10000 * EPSILON * failures
        else:
            # run the original svt multiple times to get estimates of #true positive and #false positive
            total = 10000
            true_positive, false_positive = 0, 0
            for _ in range(total):
                local_true, local_false = svt_original(all_inputs, alignments[i])
                true_positive += local_true
                false_positive += local_false
            results[i] = -true_positive / total + false_positive / total
            # results[i] = -true_positive / LENGTH + false_positive / LENGTH #variance - true_positive / LENGTH + false_negative / LENGTH #+ np.abs(cost - EPSILON) #- (count_false / LENGTH) + np.abs(count_true - N) / max(N, LENGTH - N)# - 20 * (count_true / LENGTH)
    return results


def main():
    """
    This *tries* to find the vanilla SVT, which is | eta1: 2 and eta2: 3c |, however, sometimes it finds | eta1: 5 and eta2: (2c + 2) |, which violates the privacy cost, so the LENGTH should be set higher
    also, we need to set N <= size / 5, moreover, we add a initial counterexample with dq = -1 and c = 15, to maximize the privacy cost, so that we can avoid the wrong solution.
    also, we set the bounds for eta1 to be (0, 5), so that we don't need to extend query length further
    with the above setting, we can reliably find ('alignments: eta1: 1, eta2: Omega ? 1 - q_i_dist : 0', 'eta1: 3 | eta2: 3*c')
    """
    """
    Final alignment: ('alignments: eta1: 0, eta2: Omega ? -q_i_dist : -1', 'eta1: 1 | eta2: 2*c + 6 | while: 1 * (2*c + 6)')
    """
    start = time.time()
    total_tp, total_fp, total_i = 0, 0, 0
    cex = np.array([-60 for _ in range(int(0.75 * LENGTH))] + [1000 for _ in range(int(0.1 * LENGTH))] + [50 for _ in
                                                                                                          range(
                                                                                                              int(0.15 * LENGTH))] + [
                       0 for _ in range(LENGTH)] + [0 for _ in range(LENGTH + 1)] + [0, 20])
    np.random.shuffle(cex)
    for _ in range(1000):
        tp, fp, i = svt_original(cex, np.array([1, 2, 0, 0, 0, 0, 0, 2, 0, 3, 0, 3]))
        total_tp += tp
        total_fp += fp
        total_i += i
    print(-total_tp + total_fp, total_i / 1000)
    # 12464
    total_tp, total_fp, total_i = 0, 0, 0
    for _ in range(1000):
        tp, fp, i = svt_original(cex, np.array([0, 1, 0, 0, 0, 0, -1, 1, 0, 3, 1, 5]))
        total_tp += tp
        total_fp += fp
        total_i += i
    print(-total_tp + total_fp, total_i / 1000)
    #
    print(f'Total Time: {time.time() - start}s')
