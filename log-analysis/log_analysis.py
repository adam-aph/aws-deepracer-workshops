"""
Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from datetime import datetime
from decimal import *

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from shapely.geometry.polygon import LineString
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

import cw_utils as cw


def load_data(fname):
    from os.path import isfile
    data = []

    i = 1

    while isfile('%s.%s' % (fname, i)):
        load_file('%s.%s' % (fname, i), data)
        i += 1

    load_file(fname, data)

    if i>1:
        print("Loaded %s log files (logs rolled over)" % i)

    return data


def load_file(fname, data):
    with open(fname, 'r') as f:
        for line in f.readlines():
            if "SIM_TRACE_LOG" in line:
                parts = line.split("SIM_TRACE_LOG:")[1].split('\t')[0].split(",")
                data.append(",".join(parts))


def convert_to_pandas(data, episodes_per_iteration=20):
    """
    stdout_ = 'SIM_TRACE_LOG:%d,%d,%.4f,%.4f,%.4f,%.2f,%.2f,%d,%.4f,%s,%s,%.4f,%d,%.2f,%s\n' % (
            self.episodes, self.steps, model_location[0], model_location[1], model_heading,
            self.steering_angle,
            self.speed,
            self.action_taken,
            self.reward,
            self.done,
            all_wheels_on_track,
            current_progress,
            closest_waypoint_index,
            self.track_length,
            time.time())
        print(stdout_)
    """

    df_list = list()

    # ignore the first two dummy values that coach throws at the start.
    for d in data[2:]:
        parts = d.rstrip().split(",")
        episode = int(parts[0])
        steps = int(parts[1])
        x = 100 * float(parts[2])
        y = 100 * float(parts[3])
        yaw = float(parts[4])
        steer = float(parts[5])
        throttle = float(parts[6])
        action = float(parts[7])
        reward = float(parts[8])
        done = 0 if 'False' in parts[9] else 1
        all_wheels_on_track = parts[10]
        progress = float(parts[11])
        closest_waypoint = int(parts[12])
        track_len = float(parts[13])
        tstamp = Decimal(parts[14])

        iteration = int(episode / episodes_per_iteration) + 1
        df_list.append((iteration, episode, steps, x, y, yaw, steer, throttle,
                        action, reward, done, all_wheels_on_track, progress,
                        closest_waypoint, track_len, tstamp))

    header = ['iteration', 'episode', 'steps', 'x', 'y', 'yaw', 'steer',
              'throttle', 'action', 'reward', 'done', 'on_track', 'progress',
              'closest_waypoint', 'track_len', 'timestamp']

    df = pd.DataFrame(df_list, columns=header)
    return df


def normalize_rewards(df):
    # Normalize the rewards to a 0-1 scale

    min_max_scaler = MinMaxScaler()
    scaled_vals = min_max_scaler.fit_transform(
        df['reward'].values.reshape(df['reward'].values.shape[0], 1))
    df['reward'] = pd.DataFrame(scaled_vals.squeeze())


def episode_parser(data):
    """
    Arrange data per episode
    """
    action_map = {}  # Action => [x,y,reward]
    episode_map = {}  # Episode number => [x,y,action,reward]

    for d in data[:]:
        parts = d.rstrip().split("SIM_TRACE_LOG:")[-1].split(",")
        e = int(parts[0])
        x = float(parts[2])
        y = float(parts[3])
        angle = float(parts[5])
        ttl = float(parts[6])
        action = int(parts[7])
        reward = float(parts[8])

        try:
            episode_map[e]
        except KeyError:
            episode_map[e] = np.array([0, 0, 0, 0, 0, 0])  # dummy
        episode_map[e] = np.vstack(
            (episode_map[e], np.array([x, y, action, reward, angle, ttl])))

        try:
            action_map[action]
        except KeyError:
            action_map[action] = []
        action_map[action].append([x, y, reward])

    # top laps
    total_rewards = {}
    for x in episode_map.keys():
        arr = episode_map[x]
        total_rewards[x] = np.sum(arr[:, 3])

    import operator
    top_idx = dict(sorted(total_rewards.items(),
                          key=operator.itemgetter(1),
                          reverse=True)[:])
    sorted_idx = list(top_idx.keys())

    return action_map, episode_map, sorted_idx


def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
                     edgecolor='r', alpha=0.3):
    # Create list for all the error patches
    errorboxes = []

    # Loop over data points; create box from errors at each point
    for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
        rect = Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
        errorboxes.append(rect)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)

    return 0


def v_color(ob):
    color = {
        True: '#6699cc',
        False: '#ffcc33'
    }

    return color[ob.is_simple]


def plot_coords(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, '.', color='#999999', zorder=1)


def plot_bounds(ax, ob):
    x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, '.', color='#000000', zorder=1)


def plot_line(ax, ob, color='cyan'):
    x, y = ob.xy
    ax.plot(x, y, color=color, alpha=0.7, linewidth=3, solid_capstyle='round',
            zorder=2)


def print_border(ax, waypoints, inner_border_waypoints, outer_border_waypoints,
                 color='lightgrey'):
    line = LineString(waypoints)
    plot_coords(ax, line)
    plot_line(ax, line, color)

    line = LineString(inner_border_waypoints)
    plot_coords(ax, line)
    plot_line(ax, line, color)

    line = LineString(outer_border_waypoints)
    plot_coords(ax, line)
    plot_line(ax, line, color)


def plot_top_laps(sorted_idx, episode_map, center_line, inner_border,
                  outer_border, n_laps=5):
    fig = plt.figure(n_laps, figsize=(12, n_laps * 10))
    for i in range(n_laps):
        idx = sorted_idx[i]

        episode_data = episode_map[idx]

        ax = fig.add_subplot(n_laps, 1, i + 1)

        line = LineString(center_line)
        plot_coords(ax, line)
        plot_line(ax, line)

        line = LineString(inner_border)
        plot_coords(ax, line)
        plot_line(ax, line)

        line = LineString(outer_border)
        plot_coords(ax, line)
        plot_line(ax, line)

        for idx in range(1, len(episode_data) - 1):
            x1, y1, action, reward, angle, speed = episode_data[idx]
            car_x2, car_y2 = x1 - 0.02, y1
            plt.plot([x1 * 100, car_x2 * 100], [y1 * 100, car_y2 * 100], 'b.')

    plt.show()
    plt.clf()

    return fig


def plot_evaluations(evaluations, inner, outer, graphed_value='throttle'):
    streams = evaluations.sort_values('timestamp', ascending=False).groupby('stream', sort=False)

    for name, stream in streams:
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=7.0)

        for id, episode in stream.groupby('episode'):
            plot_grid_world(episode, inner, outer, graphed_value, ax=axes[int(id / 3), id % 3])

        plt.show()
        plt.clf()


def plot_grid_world(episode_df, inner, outer, graphed_value='throttle', min_progress=None, ax=None):
    """
    plot a scaled version of lap, along with throttle taken a each position
    """

    episode_df.loc[:, 'distance_diff'] = ((episode_df['x'].shift(1) - episode_df['x']) ** 2 + (
            episode_df['y'].shift(1) - episode_df['y']) ** 2) ** 0.5

    distance = np.nansum(episode_df['distance_diff']) / 100
    lap_time = np.ptp(episode_df['timestamp'].astype(float))
    velocity = distance / lap_time
    average_throttle = np.nanmean(episode_df['throttle'])
    progress = np.nanmax(episode_df['progress'])

    if not min_progress or progress > min_progress:

        distance_lap_time = 'Distance, progress, lap time = %.2f (meters), %.2f %%, %.2f (sec)' % (
            distance, progress, lap_time)
        throttle_velocity = 'Average throttle, velocity = %.2f (Gazebo), %.2f (meters/sec)' % (
            average_throttle, velocity)

        fig = None
        if ax is None:
            fig = plt.figure(figsize=(16, 10))
            ax = fig.add_subplot(1, 1, 1)

        ax.set_facecolor('midnightblue')

        line = LineString(inner)
        plot_coords(ax, line)
        plot_line(ax, line)

        line = LineString(outer)
        plot_coords(ax, line)
        plot_line(ax, line)

        episode_df.plot.scatter('x', 'y', ax=ax, s=3, c=graphed_value, cmap=plt.get_cmap('plasma'))

        subtitle = '%s%s\n%s\n%s' % (
            ('Stream: %s, ' % episode_df['stream'].iloc[0]) if 'stream' in episode_df.columns else '',
            datetime.fromtimestamp(episode_df['timestamp'].iloc[0]),
            distance_lap_time,
            throttle_velocity)
        ax.set_title(subtitle)

        if fig:
            plt.show()
            plt.clf()


def simulation_agg(panda, firstgroup='iteration', add_timestamp=False, is_eval=False):
    grouped = panda.groupby([firstgroup, 'episode'])

    by_steps = grouped['steps'].agg(np.max).reset_index()
    by_start = grouped.first()['closest_waypoint'].reset_index() \
        .rename(index=str, columns={"closest_waypoint": "start_at"})
    by_progress = grouped['progress'].agg(np.max).reset_index()
    by_throttle = grouped['throttle'].agg(np.mean).reset_index()
    by_time = grouped['timestamp'].agg(np.ptp).reset_index() \
        .rename(index=str, columns={"timestamp": "time"})
    by_time['time'] = by_time['time'].astype(float)

    result = by_steps \
        .merge(by_start) \
        .merge(by_progress, on=[firstgroup, 'episode']) \
        .merge(by_time, on=[firstgroup, 'episode'])

    if not is_eval:
        if 'new_reward' not in panda.columns:
            print('new reward not found, using reward as its values')
            panda['new_reward'] = panda['reward']
        by_new_reward = grouped['new_reward'].agg(np.sum).reset_index()
        result = result.merge(by_new_reward, on=[firstgroup, 'episode'])

    result = result.merge(by_throttle, on=[firstgroup, 'episode'])

    if not is_eval:
        by_reward = grouped['reward'].agg(np.sum).reset_index()
        result = result.merge(by_reward, on=[firstgroup, 'episode'])

    result['time_if_complete'] = result['time'] * 100 / result['progress']

    if not is_eval:
        result['reward_if_complete'] = result['reward'] * 100 / result['progress']
        result['quintile'] = pd.cut(result['episode'], 5, labels=['1st', '2nd', '3rd', '4th', '5th'])

    if add_timestamp:
        by_timestamp = grouped['timestamp'].agg(np.max).astype(float).reset_index()
        by_timestamp['timestamp'] = pd.to_datetime(by_timestamp['timestamp'], unit='s')
        result = result.merge(by_timestamp, on=[firstgroup, 'episode'])

    return result


def scatter_aggregates(aggregate_df, title=None, is_eval=False):
    fig, axes = plt.subplots(nrows=2 if is_eval else 3, ncols=2 if is_eval else 3, figsize=[15, 11])
    if title:
        fig.suptitle(title)
    if not is_eval:
        aggregate_df.plot.scatter('time', 'reward', ax=axes[0, 2])
        aggregate_df.plot.scatter('time', 'new_reward', ax=axes[1, 2])
        aggregate_df.plot.scatter('start_at', 'reward', ax=axes[2, 2])
        aggregate_df.plot.scatter('start_at', 'progress', ax=axes[2, 0])
        aggregate_df.plot.scatter('start_at', 'time_if_complete', ax=axes[2, 1])
    aggregate_df.plot.scatter('time', 'progress', ax=axes[0, 0])
    aggregate_df.hist(column=['time'], bins=20, ax=axes[1, 0])
    aggregate_df.plot.scatter('time', 'steps', ax=axes[0, 1])
    aggregate_df.hist(column=['progress'], bins=20, ax=axes[1, 1])

    plt.show()
    plt.clf()


def analyze_categories(panda, category='quintile', groupcount=5, title=None):
    grouped = panda.groupby(category)

    fig, axes = plt.subplots(nrows=groupcount, ncols=4, figsize=[15, 15])

    if title:
        fig.suptitle(title)

    row = 0
    for name, group in grouped:
        group.plot.scatter('time', 'reward', ax=axes[row, 0])
        group.plot.scatter('time', 'new_reward', ax=axes[row, 1])
        group.hist(column=['time'], bins=20, ax=axes[row, 2])
        axes[row, 3].set(xlim=(0, 100))
        group.hist(column=['progress'], bins=20, ax=axes[row, 3])
        row += 1

    plt.show()
    plt.clf()


def analyze_training_progress(aggregates, title=None):
    aggregates['complete'] = np.where(aggregates['progress'] == 100, 1, 0)

    grouped = aggregates.groupby('iteration')

    reward_per_iteration = grouped['reward'].agg([np.mean, np.std]).reset_index()
    time_per_iteration = grouped['time'].agg([np.mean, np.std]).reset_index()
    progress_per_iteration = grouped['progress'].agg([np.mean, np.std]).reset_index()

    complete_laps = aggregates[aggregates['progress'] == 100.0]
    complete_grouped = complete_laps.groupby('iteration')

    complete_times = complete_grouped['time'].agg([np.mean, np.min, np.max]).reset_index()

    total_completion_rate = complete_laps.shape[0] / aggregates.shape[0]

    complete_per_iteration = grouped['complete'].agg([np.mean]).reset_index()

    print('Number of episodes = ', np.max(aggregates['episode']))
    print('Number of iterations = ', np.max(aggregates['iteration']))

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=[15, 15])

    if title:
        fig.suptitle(title)

    plot(axes[0, 0], reward_per_iteration, 'iteration', 'Iteration', 'mean', 'Mean reward', 'Rewards per Iteration')
    plot(axes[1, 0], reward_per_iteration, 'iteration', 'Iteration', 'std', 'Std dev of reward', 'Dev of reward')
    plot(axes[2, 0], aggregates, 'episode', 'Episode', 'reward', 'Total reward')

    plot(axes[0, 1], time_per_iteration, 'iteration', 'Iteration', 'mean', 'Mean time', 'Times per Iteration')
    plot(axes[1, 1], time_per_iteration, 'iteration', 'Iteration', 'std', 'Std dev of time', 'Dev of time')
    if complete_times.shape[0] > 0:
        plot(axes[2, 1], complete_times, 'iteration', 'Iteration', 'mean', 'Time', 'Mean completed laps time')

    plot(axes[0, 2], progress_per_iteration, 'iteration', 'Iteration', 'mean', 'Mean progress',
         'Progress per Iteration')
    plot(axes[1, 2], progress_per_iteration, 'iteration', 'Iteration', 'std', 'Std dev of progress', 'Dev of progress')
    plot(axes[2, 2], complete_per_iteration, 'iteration', 'Iteration', 'mean', 'Completion rate',
         'Completion rate (avg: %s)' % total_completion_rate)

    plt.show()
    plt.clf()


def plot(ax, df, xval, xlabel, yval, ylabel, title=None):
    df.plot.scatter(xval, yval, ax=ax, s=5, alpha=0.7)
    if title:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    plt.grid(True)


def load_eval_data(eval_fname):
    eval_data = load_data(eval_fname)
    return convert_to_pandas(eval_data)


def load_eval_logs(logs):
    full_dataframe = None
    for log in logs:
        eval_data = load_data(log[0])
        dataframe = convert_to_pandas(eval_data)
        dataframe['stream'] = log[1]
        if full_dataframe is not None:
            full_dataframe = full_dataframe.append(dataframe)
        else:
            full_dataframe = dataframe

    return full_dataframe.sort_values(
        ['stream', 'episode', 'steps']).reset_index()


def analyse_single_evaluation(log_file, inner_border, outer_border, episodes=5,
                              min_progress=None):
    eval_df = load_eval_data(log_file)

    for e in range(episodes):
        episode_df = eval_df[eval_df['episode'] == e]
        plot_grid_world(episode_df, inner_border, outer_border, min_progress=min_progress)


def analyse_multiple_race_evaluations(logs, inner_border, outer_border, min_progress=None):
    for log in logs:
        analyse_single_evaluation(log[0], inner_border, outer_border, min_progress=min_progress)


def download_and_analyse_multiple_race_evaluations(log_folder, l_inner_border, l_outer_border, not_older_than=None,
                                                   older_than=None,
                                                   log_group='/aws/deepracer/leaderboard/SimulationJobs',
                                                   min_progress=None):
    logs = cw.download_all_logs("%s/deepracer-eval-" % log_folder, log_group, not_older_than, older_than)

    analyse_multiple_race_evaluations(logs, l_inner_border, l_outer_border, min_progress=min_progress)


def df_to_params(df_row, waypoints):
    from track_utils import get_vector_length, \
        get_a_point_on_a_line_closest_to_point, is_point_on_the_line
    waypoint = df_row['closest_waypoint']
    before = waypoint - 1
    if waypoints[waypoint].tolist() == waypoints[before].tolist():
        before -= 1
    after = (waypoint + 1) % len(waypoints)

    if waypoints[waypoint].tolist() == waypoints[after].tolist():
        after = (after + 1) % len(waypoints)

    current_location = np.array([df_row['x'], df_row['y']])

    closest_point = get_a_point_on_a_line_closest_to_point(
        waypoints[before],
        waypoints[waypoint],
        [df_row['x'], df_row['y']]
    )

    if is_point_on_the_line(waypoints[before][0], waypoints[before][1],
                            waypoints[waypoint][0], waypoints[waypoint][1],
                            closest_point[0], closest_point[1]):
        closest_waypoints = [before, waypoint]
    else:
        closest_waypoints = [waypoint, after]

        closest_point = get_a_point_on_a_line_closest_to_point(
            waypoints[waypoint],
            waypoints[after],
            [df_row['x'], df_row['y']]
        )

    params = {
        'x': df_row['x'] / 100,
        'y': df_row['y'] / 100,
        'speed': df_row['throttle'],
        'steps': df_row['steps'],
        'progress': df_row['progress'],
        'heading': df_row['yaw'] * 180 / 3.14,
        'closest_waypoints': closest_waypoints,
        'steering_angle': df_row['steer'] * 180 / 3.14,
        'waypoints': waypoints / 100,
        'distance_from_center':
            get_vector_length(
                (
                        closest_point
                        -
                        current_location
                ) / 100),
        'timestamp': df_row['timestamp'],
        # TODO I didn't need them yet. DOIT
        'track_width': 0.60,
        'is_left_of_center': None,
        'all_wheels_on_track': True,
        'is_reversed': False,
    }

    return params


def new_reward(panda, center_line, reward_module, verbose=False):
    import importlib
    importlib.invalidate_caches()
    rf = importlib.import_module(reward_module)
    importlib.reload(rf)

    reward = rf.Reward(verbose=verbose)

    new_rewards = []
    for index, row in panda.iterrows():
        new_rewards.append(
            reward.reward_function(df_to_params(row, center_line)))

    panda['new_reward'] = new_rewards


def plot_track(df, center_line, inner_border, outer_border,
               track_size=(500, 800), x_shift=0, y_shift=0):
    track = np.zeros(track_size)  # lets magnify the track by *100
    for index, row in df.iterrows():
        x = int(row["x"]) + x_shift
        y = int(row["y"]) + y_shift
        reward = row["reward"]

        # clip values that are off track
        if y >= track_size[0]:
            y = track_size[0] - 1

        if x >= track_size[1]:
            x = track_size[1] - 1

        track[y, x] = reward

    fig = plt.figure(1, figsize=(12, 16))
    ax = fig.add_subplot(111)

    shifted_center_line = [[point[0] + x_shift, point[1] + y_shift] for point
                           in center_line]
    shifted_inner_border = [[point[0] + x_shift, point[1] + y_shift] for point
                            in inner_border]
    shifted_outer_border = [[point[0] + x_shift, point[1] + y_shift] for point
                            in outer_border]

    print_border(ax, shifted_center_line, shifted_inner_border,
                 shifted_outer_border)

    return track


class TrackBreakdown:
    def __init__(self, vert_lines, track_segments, segment_x, segment_y,
                 segment_xerr, segment_yerr):
        # vert_lines are indices of waypoints put on the track in squares to mark a section
        self.vert_lines = vert_lines
        # track segments determine location of descriptions on the right graph, formed of tuple
        # (location along the x axis, location along the y axis, description)
        self.track_segments = track_segments

        # marking of a bottom-left pixel of a segment on the right graph
        self.segment_x = segment_x
        self.segment_y = segment_y

        # boundaries of red rectangles on the right graph
        # how many pixels wide before and after the bottom-left pixel x coordinate
        self.segment_xerr = segment_xerr
        # how many pixels tall below and above the bottom-left pixel y coordinate
        self.segment_yerr = segment_yerr


reinvent2018 = TrackBreakdown(
    vert_lines=[10, 25, 32, 33, 40, 45, 50, 53, 61, 67],
    track_segments=[(15, 100, 'hairpin'),
                    (32, 100, 'right'),
                    (42, 100, 'left'),
                    (51, 100, 'left'),
                    (63, 100, 'left')],

    segment_x=np.array([15, 32, 42, 51, 63]),
    segment_y=np.array([0, 0, 0, 0, 0]),

    segment_xerr=np.array([[5, 1, 2, 1, 2], [10, 1, 3, 2, 4]]),
    segment_yerr=np.array([[0, 0, 0, 0, 0], [150, 150, 150, 150, 150]]))

london_loop = TrackBreakdown(
    vert_lines=[0, 15, 17, 30, 33, 45, 75, 105, 120, 132, 150, 180, 190, 210],
    track_segments=[(0, 100, 'long sharp left'),
                    (17, 90, 'mild right'),
                    (33, 80, 'tight left'),
                    (75, 100, 'mild chicane'),
                    (120, 100, 'short sharp left'),
                    (150, 90, 'left'),
                    (190, 100, 'right')],

    segment_x=np.array([0, 17, 33, 75, 120, 150, 190]),
    segment_y=np.array([0, 0, 0, 0, 0, 0, 0]),

    segment_xerr=np.array(
        [[0, 0, 0, 0, 0, 0, 0], [15, 13, 12, 30, 12, 30, 20]]),
    segment_yerr=np.array(
        [[0, 0, 0, 0, 0, 0, 0], [150, 150, 150, 150, 150, 150, 150]]))

track_breakdown = {'reinvent2018': reinvent2018, 'london_loop': london_loop}


def action_breakdown(df, iteration_ids, track_breakdown, center_line,
                     inner_border, outer_border,
                     action_names=['LEFT', 'RIGHT', 'STRAIGHT', 'SLIGHT LEFT',
                                   'SLIGHT RIGHT', 'SLOW']):
    fig = plt.figure(figsize=(16, 32))

    if type(iteration_ids) is not list:
        iteration_ids = [iteration_ids]

    wpts_array = center_line

    for iter_num in iteration_ids:
        # Slice the data frame to get all episodes in that iteration
        df_iter = df[(iter_num == df['iteration'])]
        n_steps_in_iter = len(df_iter)
        print('Number of steps in iteration=', n_steps_in_iter)

        th = 0.8
        for idx in range(len(action_names)):
            ax = fig.add_subplot(6, 2, 2 * idx + 1)
            print_border(ax, center_line, inner_border, outer_border)

            df_slice = df_iter[df_iter['reward'] >= th]
            df_slice = df_slice[df_slice['action'] == idx]

            ax.plot(df_slice['x'], df_slice['y'], 'b.')

            for idWp in track_breakdown.vert_lines:
                ax.text(wpts_array[idWp][0],
                        wpts_array[idWp][1] + 20,
                        str(idWp),
                        bbox=dict(facecolor='red', alpha=0.5))

            # ax.set_title(str(log_name_id) + '-' + str(iter_num) + ' w rew >= '+str(th))
            ax.set_ylabel(action_names[idx])

            # calculate action way point distribution
            action_waypoint_distribution = list()
            for idWp in range(len(wpts_array)):
                action_waypoint_distribution.append(
                    len(df_slice[df_slice['closest_waypoint'] == idWp]))

            ax = fig.add_subplot(6, 2, 2 * idx + 2)

            # Call function to create error boxes
            _ = make_error_boxes(ax,
                                 track_breakdown.segment_x,
                                 track_breakdown.segment_y,
                                 track_breakdown.segment_xerr,
                                 track_breakdown.segment_yerr)

            for tt in range(len(track_breakdown.track_segments)):
                ax.text(track_breakdown.track_segments[tt][0],
                        track_breakdown.track_segments[tt][1],
                        track_breakdown.track_segments[tt][2])

            ax.bar(np.arange(len(wpts_array)), action_waypoint_distribution)
            ax.set_xlabel('waypoint')
            ax.set_ylabel('# of actions')
            ax.legend([action_names[idx]])
            ax.set_ylim((0, 150))

    plt.show()
    plt.clf()

# aph: ported from Chris Thompson repo    
# Get the data for policy training based on experiences received from rollout workers
def extract_training_epochs(fname):
  df_list = list()
  tm = datetime.timestamp(datetime.now())
  with open(fname, 'r') as f:
    for line in f.readlines():
      if 'Policy training' in line:
        # Policy training> Surrogate loss=-0.08960115909576416, KL divergence=0.031318552792072296, Entropy=3.070063352584839, training epoch=3, learning_rate=0.0003\r"
        parts = line.split("Policy training> ")[1].strip().split(',')
        surrogate_loss = float(parts[0].split('Surrogate loss=')[1])
        kl_divergence = float(parts[1].split('KL divergence=')[1])
        entropy = float(parts[2].split('Entropy=')[1])
        epoch = float(parts[3].split('training epoch=')[1])
        learning_rate = float(parts[4].split('learning_rate=')[1])

        # "log_time":"2019-09-19T20:14:31+00:00"
        df_list.append((tm, surrogate_loss, kl_divergence, entropy, epoch, learning_rate))
        tm += 1

  columns = ['timestamp', 'surrogate_loss', 'kl_divergence', 'entropy', 'epoch', 'learning_rate']
  return pd.DataFrame(df_list, columns=columns).sort_values('timestamp',axis='index').reset_index(drop=True)

# Get the experience iteration summary upon receipt from rollout workers
def extract_training_iterations(fname):
  df_list = list()
  tm = datetime.timestamp(datetime.now())
  with open(fname, 'r') as f:
    for line in f.readlines():
      if 'Training' in line:
        # Training> Name=main_level/agent, Worker=0, Episode=781, Total reward=67.4, Steps=20564, Training iteration=39\r","container_id":"8963076f3fa49d5c0aac3129ba277675445e45819daf31505b41647ca2c2ec84","container_name":"/dr-training","log_time":"2019-09-20T06:02:54+00:00"}
        parts = line.split("Training> ")[1].strip().split(',')
        name = parts[0].split('Name=')[1]
        worker = int(parts[1].split('Worker=')[1])
        episode = int(parts[2].split('Episode=')[1])
        reward = float(parts[3].split('Total reward=')[1])
        steps = int(parts[4].split('Steps=')[1])
        iteration = int(parts[5].split('Training iteration=')[1])

        # "log_time":"2019-09-19T20:14:31+00:00"
        df_list.append((tm, name, worker, episode, reward, steps, iteration))
        tm += 1

  columns = ['timestamp', 'name', 'worker', 'episode', 'reward', 'steps','iteration']
  return pd.DataFrame(df_list, columns=columns).sort_values('timestamp',axis='index').reset_index(drop=True)

def plot_worker_stats(fe):
  fig = plt.figure(figsize=(16, 18))
  ax = fig.add_subplot(311)
  ax.plot(fe['timestamp'],fe['surrogate_loss'])
  ax.set_title('Surrogate Loss')
  ax = fig.add_subplot(312)
  ax.plot(fe['timestamp'],fe['kl_divergence'])
  ax.set_title('KL Divergence')
  ax = fig.add_subplot(313)
  ax.plot(fe['timestamp'],fe['entropy'])
  ax.set_title('Entropy')  

def plot_action_space_coverage(df):
  # Reject initial actions where there is no action index
  #df_not_empty = df[df['throttle'] != 0.0]
  df['degrees'] = df['steer'].transform(lambda x: math.degrees(x))
  unique_actions = df.groupby(['throttle','degrees']).size().reset_index(name='count')
  #print(df.groupby(['throttle','degrees'])

  # Plot how often model used the action choices
  fig = plt.figure(figsize=(10,12))
  ax = fig.add_subplot(211)
  sns.scatterplot(x='degrees', 
                y='throttle', 
                size='count', 
                sizes=(2,1000),
                hue='count', 
                data=unique_actions, 
                ax=ax)
  ax.set_title('Action Counts')
  ax.set_xlim(df['degrees'].max()+5,df['degrees'].min()-5);
  #pd.DataFrame(unique_actions)

  # Plot how much reward was given for action choices
  action_rewards = df.groupby(['throttle','degrees']).reward.describe().reset_index().rename(columns={'mean': 'mean reward', 'max': 'max reward'})

  ax = fig.add_subplot(212)
  sns.scatterplot(x='degrees', 
                y='throttle', 
                size='mean reward',
                sizes=(10,2000), 
                size_norm=(0,action_rewards['max reward'].max()),
                hue='mean reward',
                hue_norm=(0,action_rewards['max reward'].max()),
                legend='brief',
                data=action_rewards, 
                ax=ax)
  sns.scatterplot(x='degrees', 
                y='throttle', 
                size='max reward',
                sizes=(10,2000),
                hue='max reward',
                alpha=0.2,
                legend='brief',
                data=action_rewards, 
                ax=ax)
  ax.set_title('Mean / Max Reward Values Per Action')
  ax.set_xlim(df['degrees'].max()+5,df['degrees'].min()-5);
  return pd.DataFrame(action_rewards)

def plot_rewards_per_iteration(df, rt, tc, ei, psd):
  # Normalize the rewards to a 0-1 scale

  from sklearn.preprocessing import  MinMaxScaler
  from sklearn.preprocessing import FunctionTransformer
  min_max_scaler = MinMaxScaler()
  # Use this for normal scaled factors 0..1
  scaled_vals = min_max_scaler.fit_transform(df['reward'].values.reshape(df['reward'].values.shape[0], 1))
  # Use this for log-scale rewards
  #log_transformer = FunctionTransformer(np.log1p, validate=True)
  #scaled_vals = min_max_scaler.fit_transform(log_transformer.transform(df['reward'].values.reshape(df['reward'].values.shape[0], 1)))

  df['scaled_reward'] = pd.DataFrame(scaled_vals.squeeze())

  # reward graph per episode
  min_episode = np.min(df['episode'])
  max_episode = np.max(df['episode'])
  print('Number of episodes = ', max_episode - min_episode)

  # Gather per-episode metrics
  total_reward_per_episode = list()
  total_progress_per_episode = list()
  pace_per_episode = list()
  # dive into per-step rewards
  mean_step_reward_per_episode = list()
  min_step_reward_per_episode = list()
  max_step_reward_per_episode = list()

  for epi in range(min_episode, max_episode+1):
    df_slice = df[df['episode'] == epi]
    #    total_reward_per_episode.append(np.sum(df_slice['scaled_reward']))
    total_reward_per_episode.append(np.sum(df_slice['reward']))
    total_progress_per_episode.append(np.max(df_slice['progress']))
    elapsed_time = float(np.max(df_slice[tc])) - float(np.min(df_slice[tc]))
    pace_per_episode.append(elapsed_time * (100 / np.max(df_slice['progress'])))
    mean_step_reward_per_episode.append(np.mean(df_slice['reward']))
    min_step_reward_per_episode.append(np.min(df_slice['reward']))
    max_step_reward_per_episode.append(np.max(df_slice['reward']))

  # Generate per-iteration averages
  average_reward_per_iteration = list()
  deviation_reward_per_iteration = list()
  buffer_rew = list()
  for val in total_reward_per_episode:
    buffer_rew.append(val)
    if len(buffer_rew) == ei:
        average_reward_per_iteration.append(np.mean(buffer_rew))
        deviation_reward_per_iteration.append(np.std(buffer_rew))
        buffer_rew = list()

  average_mean_step_reward_per_iteration = list()
  deviation_mean_step_reward_per_iteration = list()
  buffer_rew = list()
  for val in mean_step_reward_per_episode:
    buffer_rew.append(val)
    if len(buffer_rew) == ei:
        average_mean_step_reward_per_iteration.append(np.mean(buffer_rew))
        deviation_mean_step_reward_per_iteration.append(np.std(buffer_rew))
        buffer_rew = list()

  average_pace_per_iteration = list()
  buffer_rew = list()
  for val in pace_per_episode:
    buffer_rew.append(val)
    if len(buffer_rew) == ei:
        average_pace_per_iteration.append(np.mean(buffer_rew))
        buffer_rew = list()

  average_progress_per_iteration = list()
  lap_completion_per_iteration = list()
  buffer_rew = list()
  for val in total_progress_per_episode:
    buffer_rew.append(val)
    if len(buffer_rew) == ei:
        average_progress_per_iteration.append(np.mean(buffer_rew))
        lap_completions = buffer_rew.count(100.0)
        lap_completion_per_iteration.append(lap_completions / ei)
        buffer_rew = list()
        
  # Plot the data
  fig = plt.figure(figsize=(16, 20))

  for rr in range(len(average_reward_per_iteration)):
    if average_reward_per_iteration[rr] >= rt :
        ax.plot(rr, average_reward_per_iteration[rr], 'r.')

  ax = fig.add_subplot(511)
  line1, = ax.plot(np.arange(len(deviation_reward_per_iteration)), deviation_reward_per_iteration)
  ax.set_ylabel('dev episode reward')
  ax.set_xlabel('Iteration')
  ax = ax.twinx()
  line2, = ax.plot(np.arange(len(deviation_mean_step_reward_per_iteration)), deviation_mean_step_reward_per_iteration, color='g')
  ax.set_ylabel('dev step reward')
  plt.legend((line1, line2), ('dev episode reward', 'dev step reward'))
  plt.grid(True)

  for rr in range(len(average_reward_per_iteration)):
    if average_reward_per_iteration[rr] >= rt:
        ax.plot(rr, deviation_reward_per_iteration[rr], 'r.')


  ax = fig.add_subplot(512)
  ax.plot(np.arange(len(total_reward_per_episode)), total_reward_per_episode, '.')
  ax.plot(np.arange(0, len(average_reward_per_iteration)*ei, ei), average_reward_per_iteration)
  ax.set_ylabel('Total Episode Rewards')
  ax.set_xlabel('Episode')
  plt.grid(True)

  ax = fig.add_subplot(513)
  ax.plot(np.arange(len(mean_step_reward_per_episode)), mean_step_reward_per_episode, '.')
  ax.plot(np.arange(0, len(average_mean_step_reward_per_iteration)*ei, ei), average_mean_step_reward_per_iteration)
  #min_reward_per_episode = list()
  #max_reward_per_episode = list()
  ax.set_ylabel('Mean Step Rewards')
  ax.set_xlabel('Episode')
  plt.grid(True)

  ax = fig.add_subplot(514)
  ax.plot(np.arange(len(total_progress_per_episode)), total_progress_per_episode, '.')
  line1, = ax.plot(np.arange(0, len(average_progress_per_iteration)*ei, ei), average_progress_per_iteration)
  ax.set_ylabel('Progress')
  ax.set_xlabel('Episode')
  ax.set_ylim(0,100)
  ax = ax.twinx()
  ax.set_ylabel('Completion Ratio')
  ax.set_ylim(0,1.0)
  line2, = ax.plot(np.arange(0, len(lap_completion_per_iteration)*ei, ei), lap_completion_per_iteration)
  ax.axhline(0.2, linestyle='--', color='r') # Target minimum 20% completion ratio
  ax.axhline(0.4, linestyle='--', color='g') # Completion of 40% should accomodate higher speeds
  plt.legend((line1, line2), ('mean progress', 'completion ratio'), loc='upper left')
  plt.grid(True)

  ax = fig.add_subplot(515)
  ax.plot(np.arange(len(pace_per_episode)), pace_per_episode, '.')
  ax.plot(np.arange(0, len(average_pace_per_iteration)*ei, ei), average_pace_per_iteration)
  ax.set_ylim(np.mean(pace_per_episode) - 2 * np.std(pace_per_episode), np.mean(pace_per_episode) + 2 * np.std(pace_per_episode))
  ax.set_ylabel('Pace')
  ax.set_xlabel('Episode')
  plt.grid(True)

  plt.show()

  # See if rewards correlate with pace
  print("Lower pace should equate to higher rewards")

  df_mean_step_reward_per_episode = pd.DataFrame(list(zip(mean_step_reward_per_episode, 
                                                        pace_per_episode,
                                                        total_progress_per_episode)), 
                                               columns=['mean_step_reward','pace', 'progress'])
  cropped = df_mean_step_reward_per_episode[abs(df_mean_step_reward_per_episode - np.mean(df_mean_step_reward_per_episode)) < psd * np.std(df_mean_step_reward_per_episode)]
  ax = cropped.plot.scatter('mean_step_reward',
                          'pace', 
                          c='progress', 
                          colormap='viridis', 
                          figsize=(12,8))
  ax.set_xlabel('Mean step reward per episode')
  ax.set_ylabel('Mean pace per episode')
