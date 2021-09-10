from dpgen import privatize
from dpgen.algorithms import sparse_vector_private, noisy_max_bq, partial_sum as partial_sum_fx, \
    smart_sum as smart_sum_fx, adaptive_svt_private, numsvt
from dpgen.frontend import ListBound
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


def main():
    privatize(
        sparse_vector,
        sparse_vector_private.main,
        privates={'q'},
        constraint=lambda q, size, t, n: n < len(q) / 5,
        original_bounds={'n': lambda q, t, n: (0, len(q))},
        related_bounds={'q': ListBound.ALL_DIFFER}
    )
    privatize(
        num_sparse_vector,
        numsvt.main,
        privates={'q'},
        constraint=lambda q, size, t, n: n < len(q) / 5,
        original_bounds={'n': lambda q, t, n: (0, len(q))},
        related_bounds={'q': ListBound.ALL_DIFFER}
    )

    privatize(
        smart_sum,
        smart_sum_fx.main,
        privates={'q'},
        constraint=lambda q, size, t, m: t < m,
        original_bounds={'m': lambda q, t, n: (0, len(q))},
        related_bounds={'q': ListBound.ONE_DIFFER}
    )

    privatize(
        partial_sum,
        partial_sum_fx.main,
        privates={'q'},
        constraint=None,
        original_bounds={},
        related_bounds={'q': ListBound.ONE_DIFFER}
    )

    privatize(
        adaptive_sparse_vector,
        adaptive_svt_private.main,
        privates={'q'},
        constraint=None,
        original_bounds={},
        related_bounds={'q': ListBound.ONE_DIFFER}
    )

    privatize(
        noisy_max,
        noisy_max_bq.main,
        privates={'q'},
        constraint=None,
        original_bounds={},
        related_bounds={'q': ListBound.ALL_DIFFER}
    )


if __name__ == '__main__':
    main()
