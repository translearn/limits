#!/usr/bin/env python

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import json
import logging
import timeit
import numpy as np
import os
import sys
import inspect
sys.path.append(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
from limit_calculation import PosVelJerkLimitation
from trajectory_plotter import TrajectoryPlotter


def denormalize(norm_value, value_range):
    actual_value = value_range[0] + 0.5 * (norm_value + 1) * (value_range[1] - value_range[0])
    return actual_value


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_step', type=float, default=0.050, help="time between network predictions "
                                                                       "(time_step = 1 / prediction_frequency)")
    parser.add_argument('--pos_limits', type=json.loads, default=None, help="pos_limits[num_joint][min/max] e.g. "
                                                                            "'[[-2.96705972839, 2.96705972839],"
                                                                            " [-2.09439510239, 2.09439510239]]'")
    parser.add_argument('--vel_limits', type=json.loads, default=None, help="vel_limits[num_joint][min/max] e.g. "
                                                                            "'[[-1.71042266695, 1.71042266695],"
                                                                            " [-1.71042266695, 1.71042266695]]'")
    parser.add_argument('--acc_limits', type=json.loads, default=None, help="acc_limits[num_joint][min/max] e.g. "
                                                                            "'[[-15, 15], [-7.5, 7.5]]'")
    parser.add_argument('--plot_joint', type=json.loads, default=None, help="whether to plot the trajectory of the "
                                                                            "corresponding joint e.g. '[1, 0]'")
    parser.add_argument('--pos_limit_factor', type=float, default=1.0, help="pos_limits are multiplied with "
                                                                            "the specified "
                                                                            "pos_limit_factor (0.0, 1.0]")
    parser.add_argument('--vel_limit_factor', type=float, default=1.0, help="vel_limits are multiplied with "
                                                                            "the specified "
                                                                            "vel_limit_factor (0.0, 1.0]")
    parser.add_argument('--acc_limit_factor', type=float, default=1.0, help="acc_limits are multiplied with "
                                                                            "the specified "
                                                                            "acc_limit_factor (0.0, 1.0]")
    parser.add_argument('--jerk_limit_factor', type=float, default=1.0, help="max_jerks are multiplied with "
                                                                             "the specified "
                                                                             "jerk_limit_factor (0.0, 1.0]")
    parser.add_argument('--trajectory_duration', type=float, default=10.0, help="duration of the generated trajectory "
                                                                                "in seconds")
    parser.add_argument('--plot_safe_acc_limits', action='store_true', default=False, help="plot the range of safe "
                                                                                           "accelerations with dashed "
                                                                                           "lines")
    parser.add_argument('--constant_action', type=float, default=None, help="a constant action [-1, 1] that "
                                                                            "is used at each decision step. If not "
                                                                            "specified, random actions are selected.")

    args = parser.parse_args()

    # beginning of user settings -------------------------------------------------------------------------------

    time_step = args.time_step

    if args.pos_limits is None:
        pos_limits = [[-2.96705972839, 2.96705972839],  # min, max Joint 1 in rad
                      [-2.09439510239, 2.09439510239],
                      [-2.96705972839, 2.96705972839],
                      [-2.09439510239, 2.09439510239],
                      [-2.96705972839, 2.96705972839],
                      [-2.09439510239, 2.09439510239],
                      [-3.05432619099, 3.05432619099]]  # min, max Joint 7 in rad
    else:
        pos_limits = args.pos_limits

    if args.vel_limits is None:
        vel_limits = [[-1.71042266695, 1.71042266695],  # min, max Joint 1 in rad/s
                      [-1.71042266695, 1.71042266695],
                      [-1.74532925199, 1.74532925199],
                      [-2.26892802759, 2.26892802759],
                      [-2.44346095279, 2.44346095279],
                      [-3.14159265359, 3.14159265359],
                      [-3.14159265359, 3.14159265359]]  # min, max Joint 7 in rad/s
    else:
        vel_limits = args.vel_limits

    if args.acc_limits is None:
        acc_limits = [[-15, 15],  # min, max Joint 1 in rad/s^2
                      [-7.5, 7.5],
                      [-10, 10],
                      [-12.5, 12.5],
                      [-15, 15],
                      [-20, 20],
                      [-20, 20]]  # min, max Joint 7 in rad/s^2
    else:
        acc_limits = args.acc_limits

    num_joints = len(pos_limits)
    if num_joints != len(vel_limits):
        raise ValueError("Expected vel_limits for {} joints but received {}".format(num_joints, len(vel_limits)))

    if num_joints != len(acc_limits):
        raise ValueError("Expected acc_limits for {} joints but received {}".format(num_joints, len(acc_limits)))

    # factors to limit the maximum position, velocity, acceleration and jerk
    # (relative to the actual limits specified below)
    pos_limit_factor = args.pos_limit_factor  # <= 1.0
    vel_limit_factor = args.vel_limit_factor  # <= 1.0
    acc_limit_factor = args.acc_limit_factor  # <= 1.0
    jerk_limit_factor = args.jerk_limit_factor  # <= 1.0

    trajectory_duration = args.trajectory_duration  # duration of the generated trajectory in seconds

    if args.plot_joint is None:
        plot_joint = [True] * num_joints  # plot all joints if not specified otherwise
    else:
        plot_joint = args.plot_joint
        if num_joints != len(plot_joint):
            raise ValueError("Expected plot_joint data for {} joints but received {}".format(num_joints,
                                                                                             len(plot_joint)))

    plot_safe_acc_limits = args.plot_safe_acc_limits
    # True: The calculated range of safe accelerations is plotted with dashed lines

    if args.constant_action is None:
        use_random_actions = True
        # if True: actions to generate the trajectory are randomly sampled
        # if False: the constant action stored in constant_action is used at each decision step
    else:
        use_random_actions = False
        constant_action = args.constant_action  # scalar within [-1, 1]

    #  end of user settings -------------------------------------------------------------------------------

    acc_limits = [[acc_limit_factor * acc_limit[0], acc_limit_factor * acc_limit[1]] for acc_limit in acc_limits]
    max_jerks = [(acc_limit[1] - acc_limit[0]) / time_step for acc_limit in acc_limits]
    jerk_limits = [[-jerk_limit_factor * max_jerk, jerk_limit_factor * max_jerk] for max_jerk in max_jerks]
    vel_limits = [[vel_limit_factor * vel_limit[0], vel_limit_factor * vel_limit[1]] for vel_limit in vel_limits]
    pos_limits = [[pos_limit_factor * pos_limit[0], pos_limit_factor * pos_limit[1]] for pos_limit in pos_limits]

    acc_limitation = PosVelJerkLimitation(time_step=time_step,
                                          pos_limits=pos_limits, vel_limits=vel_limits,
                                          acc_limits=acc_limits, jerk_limits=jerk_limits)

    trajectory_plotter = TrajectoryPlotter(time_step=time_step,
                                           pos_limits=pos_limits,
                                           vel_limits=vel_limits,
                                           acc_limits=acc_limits,
                                           jerk_limits=jerk_limits,
                                           plot_joint=plot_joint,
                                           plot_safe_acc_limits=plot_safe_acc_limits)

    current_position = [0 for _ in pos_limits]
    current_velocity = [0 for _ in vel_limits]
    current_acceleration = [0 for _ in acc_limits]

    trajectory_plotter.reset_plotter(current_position)

    trajectory_timer = timeit.default_timer()
    logging.info('Calculating trajectory ...')

    for j in range(round(trajectory_duration / time_step)):

        # calculate the range of valid actions
        safe_action_range, _ = acc_limitation.calculate_valid_acceleration_range(current_position,
                                                                                 current_velocity,
                                                                                 current_acceleration)

        # generate actions in range [-1, 1] for each joint
        # Note: Action calculation is normally performed by a neural network
        if use_random_actions:
            action = [np.random.uniform(low=-1, high=1) for _ in pos_limits]
        else:
            action = [constant_action for _ in pos_limits]

        safe_action = np.array([safe_action_range[i][0] + 0.5 * (action[i] + 1) *
                                (safe_action_range[i][1] - safe_action_range[i][0])
                                for i in range(len(action))])

        trajectory_plotter.add_data_point(safe_action, safe_action_range)

        next_acceleration = [denormalize(safe_action[k], acc_limits[k]) for k in range(len(safe_action))]

        next_position = [current_position[k] + current_velocity[k] * time_step +
                         (1 / 3 * current_acceleration[k] + 1 / 6 * next_acceleration[k]) * time_step ** 2
                         for k in range(len(current_position))]

        next_velocity = [current_velocity[k] + 0.5 * time_step * (current_acceleration[k] + next_acceleration[k])
                         for k in range(len(current_velocity))]

        current_position = next_position
        current_velocity = next_velocity
        current_acceleration = next_acceleration

    logging.info('Calculating a trajectory with a duration of ' + str(trajectory_duration) + " seconds took " +
                 str(timeit.default_timer() - trajectory_timer) + ' seconds')
    trajectory_plotter.display_plot()
