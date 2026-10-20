import numpy as np
import matplotlib.pyplot as plt

"""
To input data points from a window on the screen
Left click on mouse to input points
middle, right or return to end input
"""
def get_points() -> tuple[np.ndarray, np.ndarray]:
    plt.cla()
    plt.title('Click at least 3 points, then press Enter')
    plt.axis([0, 10, 0, 10])
    points = plt.ginput(-1, timeout=0, show_clicks=True)
    plt.ioff()
    if len(points) < 3:
        raise ValueError('Select at least 3 points before finishing')
    x, y = zip(*points)
    return np.array(x), np.array(y)