import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.path import Path


def load_track(track_name, track_path="./tracks"):
    waypoints = np.load(os.path.join(track_path, f"{track_name}.npy"))
    center_line = waypoints[:, 0:2]
    inner_border = waypoints[:, 2:4]
    outer_border = waypoints[:, 4:6]
    return center_line, inner_border, outer_border, waypoints


def calculate_polyline_length(points):
    return np.sum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))


def plot_coords(ax, ob, color, style='-'):
    x, y = ob[:, 0], ob[:, 1]
    ax.plot(x, y, '.', color=color, linestyle=style, zorder=1)


def plot_line(ax, ob, color, style='-'):
    x, y = ob[:, 0], ob[:, 1]
    ax.plot(x, y, color=color, alpha=0.7, linewidth=3, linestyle=style, solid_capstyle='round', zorder=2)


def plot_track(ax, center_line, inner_border, outer_border):
    plot_coords(ax, center_line, 'orange')
    plot_line(ax, center_line, 'orange', style='dotted')
    plot_coords(ax, inner_border, 'blue')
    plot_line(ax, inner_border, 'blue')
    plot_coords(ax, outer_border, 'green')
    plot_line(ax, outer_border, 'green')


def menger_curvature(pt1, pt2, pt3, atol=1e-3):
    vec21 = np.array([pt1[0] - pt2[0], pt1[1] - pt2[1]])
    vec23 = np.array([pt3[0] - pt2[0], pt3[1] - pt2[1]])

    norm21 = np.linalg.norm(vec21)
    norm23 = np.linalg.norm(vec23)

    theta = np.arccos(np.dot(vec21, vec23) / (norm21 * norm23))
    if np.isclose(theta - np.pi, 0.0, atol=atol):
        theta = 0.0

    dist13 = np.linalg.norm(vec21 - vec23)

    return 2 * np.sin(theta) / dist13


def point_within_polygon(point, polygon):
    return Path(polygon).contains_point(point)


def improve_race_line(old_line, inner_border, outer_border, xi_iterations=4, line_iterations=1000):
    new_line = copy.deepcopy(old_line)
    for _ in range(line_iterations):
        for i in range(len(new_line)):
            xi = new_line[i]
            npoints = len(new_line)
            prevprev = (i - 2 + npoints) % npoints
            prev = (i - 1 + npoints) % npoints
            nexxt = (i + 1 + npoints) % npoints
            nexxtnexxt = (i + 2 + npoints) % npoints
            ci = menger_curvature(new_line[prev], xi, new_line[nexxt])
            c1 = menger_curvature(new_line[prevprev], new_line[prev], xi)
            c2 = menger_curvature(xi, new_line[nexxt], new_line[nexxtnexxt])
            target_ci = (c1 + c2) / 2

            xi_bound1 = copy.deepcopy(xi)
            xi_bound2 = ((new_line[nexxt][0] + new_line[prev][0]) / 2.0, (new_line[nexxt][1] + new_line[prev][1]) / 2.0)
            p_xi = copy.deepcopy(xi)
            for _ in range(xi_iterations):
                p_ci = menger_curvature(new_line[prev], p_xi, new_line[nexxt])
                if np.isclose(p_ci, target_ci):
                    break
                if p_ci < target_ci:
                    xi_bound2 = copy.deepcopy(p_xi)
                    new_p_xi = ((xi_bound1[0] + p_xi[0]) / 2.0, (xi_bound1[1] + p_xi[1]) / 2.0)
                    if point_within_polygon(new_p_xi, inner_border) or not point_within_polygon(new_p_xi, outer_border):
                        xi_bound1 = copy.deepcopy(new_p_xi)
                    else:
                        p_xi = new_p_xi
                else:
                    xi_bound1 = copy.deepcopy(p_xi)
                    new_p_xi = ((xi_bound2[0] + p_xi[0]) / 2.0, (xi_bound2[1] + p_xi[1]) / 2.0)
                    if point_within_polygon(new_p_xi, inner_border) or not point_within_polygon(new_p_xi, outer_border):
                        xi_bound2 = copy.deepcopy(new_p_xi)
                    else:
                        p_xi = new_p_xi
            new_xi = p_xi
            new_line[i] = new_xi
    return new_line


def export_to_csv(data, filename, headers):
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(filename, index=False)


def main():
    track_name = '2022_may_open_ccw'
    center_line, inner_border, outer_border, waypoints = load_track(track_name)

    # Calculate original track length
    original_length = calculate_polyline_length(center_line)

    # Optimize the race line
    race_line = copy.deepcopy(center_line[:-1])
    race_line = improve_race_line(race_line, inner_border, outer_border)
    loop_race_line = np.append(race_line, [race_line[0]], axis=0)

    # Calculate optimized track length
    optimized_length = calculate_polyline_length(loop_race_line)

    print(f"Original track length: {original_length:.2f}")
    print(f"Optimized track length: {optimized_length:.2f}")

    # Plot original and optimized track
    fig, ax = plt.subplots(1, 2, figsize=(32, 10), facecolor='black')
    plt.axis('equal')

    # Ensure both plots have the same aspect ratio
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')

    # Plot original track
    plot_track(ax[0], center_line, inner_border, outer_border)
    ax[0].set_title('Original Track')

    # Plot optimized track
    plot_track(ax[1], center_line, inner_border, outer_border)
    plot_coords(ax[1], loop_race_line, 'red')
    plot_line(ax[1], loop_race_line, 'red')
    ax[1].set_title('Optimized Track')

    plt.show()

    # Export optimized race line to CSV
    export_to_csv(loop_race_line, 'optimized_race_line.csv', ['x', 'y'])
    # Export original waypoints to CSV
    headers = ['center_x', 'center_y', 'inner_x', 'inner_y', 'outer_x', 'outer_y']
    export_to_csv(waypoints, f'{track_name}.csv', headers)


if __name__ == '__main__':
    main()