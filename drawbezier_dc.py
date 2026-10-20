"""Draw cubic Bezier curves using de Casteljau midpoint subdivision."""
import numpy as np
import matplotlib.pyplot as plt


def subdivide(B: np.ndarray, nn: int) -> np.ndarray:
    """Return 2**nn + 1 ordered polyline vertices, shaped (2, count)."""
    B = np.asarray(B, dtype=float)
    if B.shape != (2, 4) or not np.isfinite(B).all():
        raise ValueError('B must be a finite array of shape (2, 4)')
    if not isinstance(nn, (int, np.integer)) or nn < 0:
        raise ValueError('nn must be a nonnegative integer')

    def split(control, depth):
        if depth == 0:
            return control[:, [0, 3]]
        q = (control[:, :-1] + control[:, 1:]) / 2
        r = (q[:, :-1] + q[:, 1:]) / 2
        s = (r[:, 0] + r[:, 1]) / 2
        left = np.column_stack((control[:, 0], q[:, 0], r[:, 0], s))
        right = np.column_stack((s, r[:, 1], q[:, 2], control[:, 3]))
        return np.column_stack((split(left, depth - 1)[:, :-1],
                                split(right, depth - 1)))

    return split(B, nn)


def drawbezier_dc(B: np.ndarray, nn: int, drawb: bool):
    vertices = subdivide(B, nn)
    B = np.asarray(B)
    plt.plot(vertices[0], vertices[1], linewidth=2)
    plt.plot(B[0], B[1], 'r+-' if drawb else 'r+', linewidth=0.7)
