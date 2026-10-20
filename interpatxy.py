"""Natural C2 cubic interpolation with unit parameter spacing."""
import numpy as np
import matplotlib.pyplot as plt
from bspline2b import bspline2b


def interpatxy(x: np.ndarray, y: np.ndarray):
    """Return dx, dy (N+3 entries), Bx, By (N by 4), and plot.

    Accept flat vectors or column vectors. De Boor arrays store
    d_{-1}, d_0, ..., d_{N+1} in that order.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if any(a.ndim != 1 and not (a.ndim == 2 and 1 in a.shape)
           for a in (x, y)):
        raise ValueError('x and y must be coordinate vectors')
    x, y = x.reshape(-1), y.reshape(-1)
    if len(x) != len(y) or len(x) < 3:
        raise ValueError('Provide matching x and y vectors with at least 3 points')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Coordinates must be finite')
    points = np.column_stack((x, y))
    N = len(x) - 1
    A = 4 * np.eye(N - 1)
    A += np.eye(N - 1, k=1) + np.eye(N - 1, k=-1)
    rhs = 6 * points[1:-1]
    # For N=2, both endpoint corrections modify the same row.
    rhs[0] -= points[0]
    rhs[-1] -= points[-1]
    inner = np.linalg.solve(A, rhs)
    d0 = (2 * points[0] + inner[0]) / 3
    dN = (inner[-1] + 2 * points[-1]) / 3
    d = np.vstack((points[0], d0, inner, dN, points[-1]))
    dx, dy = d[:, 0], d[:, 1]
    Bx, By = bspline2b(dx, dy, len(dx) - 1, 6, True)
    plt.plot(x, y, 'b+', markersize=9, label='Interpolation data')
    plt.legend()
    return dx, dy, Bx, By
