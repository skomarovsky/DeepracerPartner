import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_npy_to_dataframe(file_path, column_names=None):
    data = np.load(file_path, allow_pickle=True)
    if column_names is None:
        column_names = ['Waypoint_X', 'Waypoint_Y', 'Inner_X', 'Inner_Y', 'Outer_X', 'Outer_Y']
    df = pd.DataFrame(data, columns=column_names)
    return df


def export_dataframe_to_csv(df, output_path):
    df.to_csv(output_path, index=False)


def generate_csv_from_npy(file_path, output_path, column_names=None):
    df = load_npy_to_dataframe(file_path, column_names)
    export_dataframe_to_csv(df, output_path)


def classify_segments_revised(waypoints, angle_threshold=7):
    straight_segments = []
    left_turns = []
    right_turns = []

    i = 0
    while i < len(waypoints) - 1:
        prev_point = waypoints[i]
        curr_point = waypoints[i + 1]

        # Calculate vectors
        if i + 2 < len(waypoints):
            next_point = waypoints[i + 2]
        else:
            next_point = waypoints[0]

        vector1 = np.array(curr_point) - np.array(prev_point)
        vector2 = np.array(next_point) - np.array(curr_point)

        # Calculate the angle between vectors
        angle = np.degrees(np.arccos(
            np.clip(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)), -1.0, 1.0)))

        if angle >= angle_threshold:  # Threshold for considering a segment as a turn
            # Determine turn direction using the 2D cross product
            cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
            if cross_product > 0:
                left_turns.append((prev_point.tolist(), curr_point.tolist()))
            else:
                right_turns.append((prev_point.tolist(), curr_point.tolist()))
            i += 1  # Move to the next point after a turn
        else:
            straight_segments.append((prev_point.tolist(), curr_point.tolist()))
            i += 1  # Move to the next point after a straight segment

    return straight_segments, left_turns, right_turns


def analyze_and_update_segments(file_path):
    df = load_npy_to_dataframe(file_path)
    waypoints = df[['Waypoint_X', 'Waypoint_Y']].values
    straight_segments, left_turns, right_turns = classify_segments_revised(waypoints)

    segment_types = []
    for i in range(len(waypoints) - 1):
        prev_point = waypoints[i]
        curr_point = waypoints[i + 1]
        if [prev_point.tolist(), curr_point.tolist()] in [seg for seg in straight_segments]:
            segment_types.append('Straight')
        elif [prev_point.tolist(), curr_point.tolist()] in [seg for seg in left_turns]:
            segment_types.append('Left Turn')
        else:
            segment_types.append('Right Turn')

    df['Segment_Type'] = segment_types + ['Straight']  # Append 'Straight' for the last point

    return df


def plot_track(file_path, canvas=None):
    column_names = ['Waypoint_X', 'Waypoint_Y', 'Inner_X', 'Inner_Y', 'Outer_X', 'Outer_Y']
    df = load_npy_to_dataframe(file_path, column_names)

    # Extract waypoints, inner boundary, outer boundary, and center line
    waypoints = df[['Waypoint_X', 'Waypoint_Y']].values
    inner_boundary = df[['Inner_X', 'Inner_Y']].values
    outer_boundary = df[['Outer_X', 'Outer_Y']].values
    center_line = (inner_boundary + outer_boundary) / 2

    # Plot the track
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(waypoints[:, 0], waypoints[:, 1], label='Waypoints')
    ax.plot(inner_boundary[:, 0], inner_boundary[:, 1], label='Inner Boundary')
    ax.plot(outer_boundary[:, 0], outer_boundary[:, 1], label='Outer Boundary')
    ax.plot(center_line[:, 0], center_line[:, 1], 'r--', label='Center Line')

    # Highlight start and end points
    ax.scatter(waypoints[0, 0], waypoints[0, 1], color='green', label='Start', s=100)
    ax.scatter(waypoints[-1, 0], waypoints[-1, 1], color='red', label='End', s=100)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()
    ax.set_title(f'DeepRacer Track: {file_path.split("/")[-1]}')

    if canvas:
        canvas.figure = fig
        canvas.draw()
    else:
        plt.show()


def plot_track_segments(file_path, canvas=None):
    df = analyze_and_update_segments(file_path)
    waypoints = df[['Waypoint_X', 'Waypoint_Y']].values
    inner_boundary = df[['Inner_X', 'Inner_Y']].values
    outer_boundary = df[['Outer_X', 'Outer_Y']].values
    center_line = (inner_boundary + outer_boundary) / 2

    # Classify segments
    straight_segments, left_turns, right_turns = classify_segments_revised(waypoints)

    # Plot the track
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(waypoints[:, 0], waypoints[:, 1], label='Waypoints')
    ax.plot(inner_boundary[:, 0], inner_boundary[:, 1], label='Inner Boundary')
    ax.plot(outer_boundary[:, 0], outer_boundary[:, 1], label='Outer Boundary')
    ax.plot(center_line[:, 0], center_line[:, 1], 'r--', label='Center Line')

    # Plot segments with different colors
    for segment in straight_segments:
        ax.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 'g-', linewidth=2)
    for segment in left_turns:
        ax.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 'b-', linewidth=2)
    for segment in right_turns:
        ax.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 'r-', linewidth=2)

    # Highlight start and end points
    ax.scatter(waypoints[0, 0], waypoints[0, 1], color='green', label='Start', s=100)
    ax.scatter(waypoints[-1, 0], waypoints[-1, 1], color='red', label='End', s=100)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()
    ax.set_title(f'DeepRacer Track Segments: {file_path.split("/")[-1]}')
    ax.grid(True)

    if canvas:
        canvas.figure = fig
        canvas.draw()
    else:
        plt.show()


def save_track_to_csv(file_path, output_path):
    df = load_npy_to_dataframe(file_path)
    export_dataframe_to_csv(df, output_path)


def save_segments_to_csv(file_path, output_path):
    df = analyze_and_update_segments(file_path)
    export_dataframe_to_csv(df, output_path)