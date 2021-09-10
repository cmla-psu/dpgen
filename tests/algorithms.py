from dpgen.frontend.annotation import output, is_private


def noisy_max(q: list[float], size: int):
    i, bq, imax = 0, 0, 0
    while i < size:
        if q[i] > bq or i == 0:
            imax = i
            bq = q[i]
        i = i + 1
    output(imax)


def sparse_vector(q: list[float], size: int, t: float, n: int):
    i, count = 0, 0
    while i < size and count < n:
        if q[i] >= t:
            output(True)
            count += 1
        else:
            output(False)
        i = i + 1


def sparse_vector_inverse(q: list[float], size: int, t: float, n: int):
    i, count = 0, 0
    while i < size and count < n:
        if q[i] >= t:
            output(True)
        else:
            output(False)
            count += 1
        i = i + 1


def adaptive_sparse_vector(q: list[float], size: int, t: float, sigma: float):
    i = 0
    while is_private and i < size:
        if q[i] - t >= sigma:
            output(q[i] - t)
        else:
            if q[i] - t >= 0:
                output(q[i] - t)
            else:
                output(0)
        i = i + 1


def gap_sparse_vector(q: list[float], size: int, t: float, n: int):
    i, count = 0, 0
    while i < size and count < n:
        if q[i] >= t:
            output(q[i] - t)
            count += 1
        else:
            output(False)
        i = i + 1


def num_sparse_vector(q: list[float], size: int, t: float, n: int):
    i, count = 0, 0
    while i < size and count < n:
        if q[i] >= t:
            output(q[i])
            count += 1
        else:
            output(False)
        i = i + 1


def partial_sum(q: list[float], size: int):
    i, vsum = 0, 0
    while i < size:
        vsum = vsum + q[i]
        i = i + 1
    output(vsum)


def smart_sum(q: list[float], size: int, t: int, m: int):
    i, vnext, vsum = 0, 0, 0
    while i < size and i <= t:
        if (i + 1) % m == 0:
            vnext = vsum + q[i]
            vsum = 0
            output(vnext)
        else:
            vnext = vnext + q[i]
            vsum = vsum + q[i]
            output(vnext)
        i = i + 1
