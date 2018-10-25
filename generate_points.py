import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate

start = 0
end = 570

angles = [90, 180, 270, 360]
samples_per_class = 3000

def rotate_point(point, angle):
    """Rotate two point by an angle.

    Parameters
    ----------
    point: 2d numpy array
        The coordinate to rotate.
    angle: float
        The angle of rotation of the point, in degrees.

    Returns
    -------
    2d numpy array
        Rotated point.
    """
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_point = rotation_matrix.dot(point)
    return rotated_point

def generate_spiral(samples, start, end, noise=0.5):
    """Generate a spiral of points.

    Given a starting end, an end angle and a noise factor, generate a spiral of points along
    an arc.

    Parameters
    ----------
    samples: int
        Number of points to generate.
    start: float
        The starting angle of the spiral in degrees.
    end: float
        The end angle at which to rotate the points, in degrees.
    noise: float
        The noisyness of the points inside the spirals. Needs to be less than 1.
    """
    # Generate some points from the square root of random data inside an uniform distribution on [0, 1).
    points = math.radians(start) + np.sqrt(np.random.rand(samples, 1)) * math.radians(end)

    # Apply a rotation to the points.
    rotated_x_axis = np.cos(points) * points + np.random.rand(samples, 1) * noise
    rotated_y_axis = np.sin(points) * points + np.random.rand(samples, 1) * noise

    # Stack the vectors inside a samples x 2 matrix.
    return np.column_stack((rotated_x_axis, rotated_y_axis))

for i, angle in enumerate(angles):
    points = generate_spiral(samples_per_class, start, end)
    rotated_points = np.apply_along_axis(rotate_point, 1, points, math.radians(angle))

    plt.plot(rotated_points[:, 0], rotated_points[:, 1], '.', str(i))

plt.show()
