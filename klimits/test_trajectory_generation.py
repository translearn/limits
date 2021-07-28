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

from klimits import PosVelJerkLimitation
from trajectory_plotter import TrajectoryPlotter
from klimits import denormalize
from klimits import calculate_end_position as calculate_end_position
from klimits import calculate_end_velocity as calculate_end_velocity
from klimits import get_num_threads


def test_trajectory_generation(time_step, pos_limits, vel_limits, acc_limits, pos_limit_factor, vel_limit_factor,
                               acc_limit_factor, jerk_limit_factor, trajectory_duration,
                               constant_action=None, num_threads=1, plot_joint=None, no_plot=False,
                               plot_safe_acc_limits=False, seed=None, return_summary=False):
    acc_limits = [[acc_limit_factor * acc_limit[0], acc_limit_factor * acc_limit[1]] for acc_limit in acc_limits]
    max_jerks = [(acc_limit[1] - acc_limit[0]) / time_step for acc_limit in acc_limits]
    jerk_limits = [[-jerk_limit_factor * max_jerk, jerk_limit_factor * max_jerk] for max_jerk in max_jerks]
    vel_limits = [[vel_limit_factor * vel_limit[0], vel_limit_factor * vel_limit[1]] for vel_limit in vel_limits]
    pos_limits = [[pos_limit_factor * pos_limit[0], pos_limit_factor * pos_limit[1]] for pos_limit in pos_limits]

    if constant_action is None:
        use_random_actions = True
        # if True: actions to generate the trajectory are randomly sampled
        # if False: the constant action stored in constant_action is used at each decision step
        if seed is not None:
            np.random.seed(seed)
    else:
        use_random_actions = False

    if num_threads is None:
        logging.info("Using %s thread(s) to compute the range of safe accelerations.",
                     get_num_threads())
    else:
        logging.info("Using %s thread(s) to compute the range of safe accelerations.", num_threads)
    logging.info("Note: The best performance is usually achieved by setting either --num_threads or OMP_NUM_THREADS to "
                 "the number of physical (not virtual) CPU cores available on your system.")

    acc_limitation = PosVelJerkLimitation(time_step=time_step,
                                          pos_limits=pos_limits, vel_limits=vel_limits,
                                          acc_limits=acc_limits, jerk_limits=jerk_limits,
                                          acceleration_after_max_vel_limit_factor=0.0001,
                                          normalize_acc_range=False, num_threads=num_threads)

    num_joints = len(pos_limits)
    current_position = np.zeros(num_joints)
    current_velocity = np.zeros(num_joints)
    current_acceleration = np.zeros(num_joints)

    if not no_plot or return_summary:
        if plot_joint is None:
            plot_joint = [True] * num_joints  # plot all joints if not specified otherwise
        else:
            if num_joints != len(plot_joint):
                raise ValueError("Expected plot_joint data for {} joints but received {}".format(num_joints,
                                                                                                 len(plot_joint)))
        trajectory_plotter = TrajectoryPlotter(time_step=time_step,
                                               pos_limits=pos_limits,
                                               vel_limits=vel_limits,
                                               acc_limits=acc_limits,
                                               jerk_limits=jerk_limits,
                                               plot_joint=plot_joint,
                                               plot_safe_acc_limits=plot_safe_acc_limits,
                                               plot_violation=False)

        trajectory_plotter.reset_plotter(current_position)

    if not use_random_actions:
        action = np.full(shape=len(pos_limits), fill_value=constant_action)
    logging.info("Calculating trajectory ...")
    trajectory_start_timer = timeit.default_timer()

    for j in range(round(trajectory_duration / time_step)):

        # calculate the range of valid actions
        safe_action_range, violation_code = acc_limitation.calculate_valid_acceleration_range(current_position,
                                                                                              current_velocity,
                                                                                              current_acceleration,
                                                                                              time_step_counter=j)
        # generate actions in range [-1, 1] for each joint
        # Note: Action calculation is normally performed by a neural network
        if use_random_actions:
            action = np.random.uniform(low=-1, high=1, size=len(pos_limits))

        next_acceleration = denormalize(action, safe_action_range.T)

        if not no_plot or return_summary:
            trajectory_plotter.add_data_point(next_acceleration, safe_action_range, violation_code)

        next_position = calculate_end_position(current_acceleration, next_acceleration, current_velocity,
                                               current_position, time_step)
        next_velocity = calculate_end_velocity(current_acceleration, next_acceleration, current_velocity, time_step)

        current_position = next_position
        current_velocity = next_velocity
        current_acceleration = next_acceleration

    trajectory_end_timer = timeit.default_timer()
    logging.info("Calculating a trajectory with a duration of %s s and a time step of %s s took %s s for %s joints.",
                 trajectory_duration, time_step, trajectory_end_timer - trajectory_start_timer, num_joints)
    if not no_plot:
        trajectory_plotter.display_plot()

    if return_summary:
        return trajectory_plotter.get_trajectory_summary()


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_step', type=float, default=0.050, help="time between network predictions "
                                                                       "(time_step = 1 / prediction_frequency)"
                                                                       ", default: %(default)s")
    parser.add_argument('--pos_limits', type=json.loads, default=None, help="pos_limits[num_joint][min/max] e.g. "
                                                                            "'[[-2.96705972839, 2.96705972839],"
                                                                            " [-2.09439510239, 2.09439510239]]'")
    parser.add_argument('--vel_limits', type=json.loads, default=None, help="vel_limits[num_joint][min/max] e.g. "
                                                                            "'[[-1.71042266695, 1.71042266695],"
                                                                            " [-1.71042266695, 1.71042266695]]'")
    parser.add_argument('--acc_limits', type=json.loads, default=None, help="acc_limits[num_joint][min/max] e.g. "
                                                                            "'[[-15, 15], [-7.5, 7.5]]'")
    parser.add_argument('--pos_limit_factor', type=float, default=1.0, help="pos_limits are multiplied with "
                                                                            "the specified "
                                                                            "pos_limit_factor (0.0, 1.0], "
                                                                            "default: %(default)s")
    parser.add_argument('--vel_limit_factor', type=float, default=1.0, help="vel_limits are multiplied with "
                                                                            "the specified "
                                                                            "vel_limit_factor (0.0, 1.0], "
                                                                            "default: %(default)s")
    parser.add_argument('--acc_limit_factor', type=float, default=1.0, help="acc_limits are multiplied with "
                                                                            "the specified "
                                                                            "acc_limit_factor (0.0, 1.0], "
                                                                            "default: %(default)s")
    parser.add_argument('--jerk_limit_factor', type=float, default=1.0, help="max_jerks are multiplied with "
                                                                             "the specified "
                                                                             "jerk_limit_factor (0.0, 1.0], "
                                                                             "default: %(default)s")
    parser.add_argument('--trajectory_duration', type=float, default=10.0, help="duration of the generated trajectory "
                                                                                "in seconds, default: %(default)s")
    parser.add_argument('--constant_action', type=float, default=None, help="a constant action [-1, 1] that "
                                                                            "is used at each decision step. If not "
                                                                            "specified, random actions are selected")
    parser.add_argument('--plot_joint', type=json.loads, default=None, help="whether to plot the trajectory of the "
                                                                            "corresponding joint e.g. '[1, 0]'")
    parser.add_argument('--plot_safe_acc_limits', action='store_true', default=False, help="plot the range of safe "
                                                                                           "accelerations with dashed "
                                                                                           "lines")
    parser.add_argument('--no_plot', action='store_true', default=False, help="if set, no plot is generated")
    parser.add_argument('--num_threads', type=int, default=None, help="the number of threads used for parallel "
                                                                      "execution. If not specified, the number of "
                                                                      "threads is either determined by the environment "
                                                                      "variable OMP_NUM_THREADS or by the number of "
                                                                      "logical CPU cores available on your system")
    parser.add_argument('--seed', type=int, default=None, help="seed the generator of random actions with an integer "
                                                               "(for debugging purposes)")

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
        acc_limits = [[-15.0, 15.0],  # min, max Joint 1 in rad/s^2
                      [-7.5, 7.5],
                      [-10.0, 10.0],
                      [-12.5, 12.5],
                      [-15.0, 15.0],
                      [-20.0, 20.0],
                      [-20.0, 20.0]]  # min, max Joint 7 in rad/s^2
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

    plot_joint = args.plot_joint

    plot_safe_acc_limits = args.plot_safe_acc_limits
    # True: The calculated range of safe accelerations is plotted with dashed lines

    constant_action = args.constant_action  # scalar within [-1, 1] or None -> random actions
    seed = args.seed  # if not None: seed of the random number generator
    num_threads = args.num_threads  # number of threads to use for the computation
    no_plot = args.no_plot  # whether to plot the generated trajectory

    #  end of user settings -------------------------------------------------------------------------------

    test_trajectory_generation(time_step=time_step, pos_limits=pos_limits, vel_limits=vel_limits,
                               acc_limits=acc_limits, pos_limit_factor=pos_limit_factor,
                               vel_limit_factor=vel_limit_factor, acc_limit_factor=acc_limit_factor,
                               jerk_limit_factor=jerk_limit_factor, trajectory_duration=trajectory_duration,
                               constant_action=constant_action,
                               num_threads=num_threads, plot_joint=plot_joint,
                               no_plot=no_plot, plot_safe_acc_limits=plot_safe_acc_limits,
                               seed=seed, return_summary=False)
