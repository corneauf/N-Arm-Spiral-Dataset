import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate

start = 0
end = 360

multipliers = [(-1, 1), (1, -1)]
samples_per_class = 5000

def generate_spiral(samples, start, end, noise=0.5, x_axis_multiplier=1, y_axis_multiplier=1):
    """Generate a spiral of points.

    Given a starting end, an end angle and a noise factor, generate a spiral of points along
    an arc.

    Parameters
    ----------
    samples: int
        Number of points to generate.
    start: float
        The starting end of the spiral in degrees.
    end: float
        The end at which to rotate the points, in degrees.
    noise: float
        The noisyness of the points inside the spirals. Needs to be less than 1.
    """
    # Generate some points from the square root of random data inside an uniform distribution on [0, 1).
    points = math.radians(start) + np.sqrt(np.random.rand(samples, 1)) * math.radians(end)

    # Apply a rotation to the points.
    rotated_x_axis = x_axis_multiplier * np.cos(points) * points + np.random.rand(samples, 1) * noise
    rotated_y_axis = y_axis_multiplier * np.sin(points) * points + np.random.rand(samples, 1) * noise

    # Stack the vectors inside a samples x 2 matrix.
    return np.column_stack((rotated_x_axis, rotated_y_axis))

for i, (x_axis, y_axis) in enumerate(multipliers):
    rotated_points = generate_spiral(samples_per_class, start, end, x_axis_multiplier=x_axis, y_axis_multiplier=y_axis)

    plt.plot(rotated_points[:, 0], rotated_points[:, 1], '.', str(i))

plt.show()
