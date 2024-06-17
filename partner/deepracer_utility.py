import numpy as np
import matplotlib.pyplot as plt


def plot_track(file_path, canvas):
    data = np.load(file_path, allow_pickle=True)
    waypoints = data[:, 0:2]
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'o-')
    canvas.draw()


def plot_track_segments(file_path, canvas):
    data = np.load(file_path, allow_pickle=True)
    waypoints = data[:, 0:2]
    straight_segments, left_turns, right_turns = classify_segments_revised(waypoints)

    for segment in straight_segments:
        plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 'g-')
    for segment in left_turns:
        plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 'b-')
    for segment in right_turns:
        plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 'r-')

    canvas.draw()


def classify_segments_revised(waypoints, angle_threshold=7):
    straight_segments = []
    left_turns = []
    right_turns = []

    i = 0
    while i < len(waypoints) - 1:
        prev_point = waypoints[i]
        curr_point = waypoints[i + 1]

        if i + 2 < len(waypoints):
            next_point = waypoints[i + 2]
        else:
            next_point = waypoints[0]

        vector1 = np.array(curr_point) - np.array(prev_point)
        vector2 = np.array(next_point) - np.array(curr_point)

        if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
            angle = 0
        else:
            angle = np.degrees(np.arccos(
                np.clip(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)), -1.0, 1.0)))

        if angle >= angle_threshold:
            cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
            if cross_product > 0:
                left_turns.append((prev_point.tolist(), curr_point.tolist()))
            else:
                right_turns.append((prev_point.tolist(), curr_point.tolist()))
            i += 1
        else:
            straight_segments.append((prev_point.tolist(), curr_point.tolist()))
            i += 1

    return straight_segments, left_turns, right_turns


def save_track_to_csv(file_path, output_path):
    data = np.load(file_path, allow_pickle=True)
    np.savetxt(output_path, data, delimiter=",")


def save_segments_to_csv(file_path, output_path):
    data = np.load(file_path, allow_pickle=True)
    waypoints = data[:, 0:2]
    straight_segments, left_turns, right_turns = classify_segments_revised(waypoints)

    with open(output_path, 'w') as f:
        f.write('Segment Type,Start Point X,Start Point Y,End Point X,End Point Y\n')

        for segment in straight_segments:
            f.write(f'Straight,{segment[0][0]},{segment[0][1]},{segment[1][0]},{segment[1][1]}\n')
        for segment in left_turns:
            f.write(f'Left Turn,{segment[0][0]},{segment[0][1]},{segment[1][0]},{segment[1][1]}\n')
        for segment in right_turns:
            f.write(f'Right Turn,{segment[0][0]},{segment[0][1]},{segment[1][0]},{segment[1][1]}\n')