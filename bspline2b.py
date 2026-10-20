"""Convert clamped uniform cubic de Boor points to Bezier segments."""
import numpy as np
import matplotlib.pyplot as plt
from drawbezier_dc import drawbezier_dc


def bspline2b(dx, dy, N: int, nn: int, drawb: bool):
    """N is the last array index; there are N - 2 Bezier segments.

    Array index j stores mathematical point d_{j-1}. Each output row
    contains four Bezier control coordinates.
    """
    dx = np.asarray(dx, dtype=float).reshape(-1)
    dy = np.asarray(dy, dtype=float).reshape(-1)
    if N < 4 or len(dx) != N + 1 or len(dy) != N + 1:
        raise ValueError('Expected N >= 4 and N + 1 de Boor points')
    if not np.isfinite(dx).all() or not np.isfinite(dy).all():
        raise ValueError('Control points must be finite')
    d = np.column_stack((dx, dy))
    count = N - 2
    control = np.empty((count, 4, 2))
    control[0, 1] = d[1]
    control[0, 2] = (d[1] + d[2]) / 2
    for i in range(1, count - 1):
        control[i, 1] = (2 * d[i + 1] + d[i + 2]) / 3
        control[i, 2] = (d[i + 1] + 2 * d[i + 2]) / 3
    control[-1, 1] = (d[-3] + d[-2]) / 2
    control[-1, 2] = d[-2]
    control[0, 0] = d[0]
    control[-1, 3] = d[-1]
    # The join bisects adjacent handles for matching first derivatives.
    joins = (control[:-1, 2] + control[1:, 1]) / 2
    control[:-1, 3] = joins
    control[1:, 0] = joins
    Bx, By = control[:, :, 0], control[:, :, 1]
    plt.figure()
    plt.plot(dx, dy, 'or--', linewidth=0.7, markersize=4,
             label='de Boor control polygon')
    for segment in control:
        drawbezier_dc(segment.T, nn, drawb)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.25)
    return Bx, By
