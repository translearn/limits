# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import datetime
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from klimits import normalize as normalize
from klimits import normalize_batch as normalize_batch
from klimits import interpolate_position_batch as interpolate_position_batch
from klimits import interpolate_velocity_batch as interpolate_velocity_batch
from klimits import interpolate_acceleration_batch as interpolate_acceleration_batch


class TrajectoryPlotter:
    def __init__(self,
                 time_step=None,
                 control_time_step=None,
                 pos_limits=None,
                 vel_limits=None,
                 acc_limits=None,
                 jerk_limits=None,
                 plot_joint=None,
                 plot_safe_acc_limits=False,
                 plot_time_limits=None,
                 plot_violation=False):

        self._time_step = time_step
        self._control_time_step = control_time_step
        self._plot_safe_acc_limits = plot_safe_acc_limits
        self._plot_time_limits = plot_time_limits
        self._plot_violation = plot_violation

        self._plot_num_sub_time_steps = max(int(1000 * time_step), 2)
        self._time_step_counter = None
        self._current_jerk = None
        self._current_acc = None
        self._current_vel = None
        self._current_pos = None

        self._time = None
        self._pos = None
        self._vel = None
        self._acc = None
        self._jerk = None
        self._violation = None

        self._sub_time = None
        self._sub_pos = None
        self._sub_vel = None
        self._sub_acc = None
        self._sub_jerk = None

        self._pos_limits = np.array(pos_limits)
        self._pos_limits_min_max = np.swapaxes(self._pos_limits, 0, 1)
        self._vel_limits = np.array(vel_limits)
        self._vel_limits_min_max = np.swapaxes(self._vel_limits, 0, 1)
        self._acc_limits = np.array(acc_limits)
        self._acc_limits_min_max = np.swapaxes(self._acc_limits, 0, 1)
        self._jerk_limits = np.array(jerk_limits)
        self._jerk_limits_min_max = np.swapaxes(self._jerk_limits, 0, 1)

        self._num_joints = len(self._pos_limits)

        if plot_joint is None:
            self._plot_joint = [True for _ in range(self._num_joints)]
        else:
            self._plot_joint = plot_joint

        self._episode_counter = 0
        self._zero_vector = np.zeros(self._num_joints)

        self._current_acc_limits = None

        self._timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        matplotlib.rcParams['text.usetex'] = False

    @property
    def trajectory_time(self):
        return self._time[-1]

    def reset_plotter(self, initial_joint_position):

        self._time_step_counter = 0
        self._episode_counter = self._episode_counter + 1

        self._current_acc = self._zero_vector.copy()
        self._current_vel = self._zero_vector.copy()
        self._current_pos = np.array(initial_joint_position.copy())

        self._pos = []
        self._vel = []
        self._acc = []
        self._jerk = []
        self._violation = []
        self._current_acc_limits = []
        self._current_acc_limits.append([[0, 0] for _ in range(self._num_joints)])

        self._time = [0]

        self._pos.append(normalize(self._current_pos, self._pos_limits_min_max))
        self._vel.append(normalize(self._current_vel, self._vel_limits_min_max))
        self._acc.append(normalize(self._current_acc, self._acc_limits_min_max))
        self._jerk.append(self._zero_vector.copy())

        self._sub_time = [0]
        self._sub_pos = self._pos.copy()
        self._sub_vel = self._vel.copy()
        self._sub_acc = self._acc.copy()
        self._sub_jerk = self._jerk.copy()

    def display_plot(self, max_time=None):

        num_subplots = 4
        if self._plot_violation:
            ax_violation = num_subplots
            num_subplots = num_subplots + 1
        else:
            ax_violation = None
        fig, ax = plt.subplots(num_subplots, 1, sharex=True)
        plt.subplots_adjust(left=0.05, bottom=0.04, right=0.95, top=0.98, wspace=0.15, hspace=0.15)
        ax_pos = 0
        ax_vel = 1
        ax_acc = 2
        ax_jerk = 3

        for i in range(len(ax)):
            ax[i].grid(True)
            ax[i].set_xlabel('Time [s]')

        if ax_pos is not None:
            ax[ax_pos].set_ylabel('Position')

        if ax_vel is not None:
            ax[ax_vel].set_ylabel('Velocity')

        if ax_jerk is not None:
            ax[ax_jerk].set_ylabel('Jerk')

        if ax_acc is not None:
            ax[ax_acc].set_ylabel('Acceleration')

        if ax_violation is not None:
            ax[ax_violation].set_ylabel('Ignored')
            ax[ax_violation].set_yticks(np.arange(5))
            ax[ax_violation].set_yticklabels(['None', 'Jerk', 'Vel', 'Pos', 'Acc'])

        joint_pos = np.swapaxes(self._pos, 0, 1)
        joint_vel = np.swapaxes(self._vel, 0, 1)
        joint_acc = np.swapaxes(self._acc, 0, 1)
        joint_jerk = np.swapaxes(self._jerk, 0, 1)
        joint_violation = np.swapaxes(self._violation, 0, 1)

        if self._plot_safe_acc_limits:
            joint_acc_limits = np.swapaxes(self._current_acc_limits, 0, 1)
            joint_acc_limits = np.swapaxes(joint_acc_limits, 1, 2)

        joint_sub_pos = np.swapaxes(self._sub_pos, 0, 1)
        joint_sub_vel = np.swapaxes(self._sub_vel, 0, 1)
        joint_sub_acc = np.swapaxes(self._sub_acc, 0, 1)
        joint_sub_jerk = np.swapaxes(self._sub_jerk, 0, 1)

        if self._plot_time_limits is not None:
            self._time = np.asarray(self._time) - self._plot_time_limits[0]
            self._sub_time = np.asarray(self._sub_time) - self._plot_time_limits[0]

        if max_time is None or max_time >= self._time[-1]:
            time_max_index = len(self._time)
        else:
            time_max_index = np.argmin(np.asarray(self._time) <= max_time)
        if max_time is None or max_time >= self._sub_time[-1]:
            sub_time_max_index = len(self._sub_time)
        else:
            sub_time_max_index = np.argmin(np.asarray(self._sub_time) <= max_time)

        for j in range(self._num_joints):
            logging.info('Joint ' + str(j + 1) + ' (min/max)' +
                         ' Jerk: ' + str(np.min(joint_sub_jerk[j])) + ' / ' + str(np.max(joint_sub_jerk[j])) +
                         '; Acc: ' + str(np.min(joint_sub_acc[j])) + ' / ' + str(np.max(joint_sub_acc[j])) +
                         '; Vel: ' + str(np.min(joint_sub_vel[j])) + ' / ' + str(np.max(joint_sub_vel[j])) +
                         '; Pos: ' + str(np.min(joint_sub_pos[j])) + ' / ' + str(np.max(joint_sub_pos[j])))

        linestyle = '-'
        for j in range(self._num_joints):
            color = 'C' + str(j)  
            color_limits = 'C' + str(j)
            marker = '.'

            if self._plot_joint[j]:
                label = 'Joint ' + str(j + 1)
                if ax_pos is not None:
                    ax[ax_pos].plot(self._time[:time_max_index], joint_pos[j][:time_max_index], color=color,
                                    marker=marker, linestyle='None', label='_nolegend_')
                    ax[ax_pos].plot(self._sub_time[:sub_time_max_index], joint_sub_pos[j][:sub_time_max_index],
                                    color=color, linestyle=linestyle, label=label)

                if ax_vel is not None:
                    ax[ax_vel].plot(self._time[:time_max_index], joint_vel[j][:time_max_index], color=color,
                                    marker=marker, linestyle='None', label='_nolegend_')
                    ax[ax_vel].plot(self._sub_time[:sub_time_max_index], joint_sub_vel[j][:sub_time_max_index],
                                    color=color, linestyle=linestyle, label=label)

                if ax_acc is not None:
                    ax[ax_acc].plot(self._time[:time_max_index], joint_acc[j][:time_max_index], color=color,
                                    marker=marker, linestyle='None', label='_nolegend_')
                    ax[ax_acc].plot(self._sub_time[:sub_time_max_index], joint_sub_acc[j][:sub_time_max_index],
                                    color=color, linestyle=linestyle, label=label)

                    if self._plot_safe_acc_limits:
                        for i in range(2):
                            ax[ax_acc].plot(self._time[:time_max_index], joint_acc_limits[j][i][:time_max_index],
                                            color=color_limits, linestyle='--', label='_nolegend_')

                if ax_jerk is not None:
                    ax[ax_jerk].plot(self._time[:time_max_index], joint_jerk[j][:time_max_index], color=color,
                                     marker=marker, linestyle='None', label='_nolegend_')
                    ax[ax_jerk].plot(self._sub_time[:sub_time_max_index], joint_sub_jerk[j][:sub_time_max_index],
                                     color=color, linestyle=linestyle, label=label)

                if ax_violation is not None:
                    ax[ax_violation].plot(self._time[:time_max_index-1], joint_violation[j][:time_max_index],
                                          color=color, linestyle=linestyle, label=label)

        for i in range(len(ax)):
            if self._plot_time_limits is None:
                ax[i].set_xlim([0, self._time[-1]])  
            else:
                ax[i].set_xlim([0, self._plot_time_limits[1] - self._plot_time_limits[0]])

        ax[-1].legend(loc='lower right')

        if ax_acc is not None:
            ax[ax_acc].set_ylim([-1.05, 1.05])
        if ax_jerk is not None:
            ax[ax_jerk].set_ylim([-1.05, 1.05])

        fig.align_ylabels(ax)
        fig.set_size_inches((24.1, 13.5), forward=False)
        plt.show()

    def add_data_point(self, current_acc, acc_range=None, violation_code=None):
        self._time.append(self._time[-1] + self._time_step)
        last_acc = self._current_acc.copy()
        last_vel = self._current_vel.copy()
        last_pos = self._current_pos.copy()
        self._current_acc = current_acc
        self._current_jerk = (self._current_acc - last_acc) / self._time_step
        self._jerk.append(normalize(self._current_jerk, self._jerk_limits_min_max))
        self._violation.append(violation_code)

        self._current_acc_limits.append(normalize_batch(acc_range.T, self._acc_limits_min_max).T)

        for j in range(1, self._plot_num_sub_time_steps + 1):
            self._sub_jerk.append(self._jerk[-1])
        time_since_start = np.linspace(self._time_step / self._plot_num_sub_time_steps, self._time_step,
                                       self._plot_num_sub_time_steps)
        self._sub_time.extend(list(time_since_start + self._time_step_counter * self._time_step))
        sub_current_acc = interpolate_acceleration_batch(last_acc, self._current_acc, time_since_start, self._time_step)
        sub_current_vel = interpolate_velocity_batch(last_acc, self._current_acc, last_vel, time_since_start,
                                                     self._time_step)
        sub_current_pos = interpolate_position_batch(last_acc, self._current_acc, last_vel, last_pos, time_since_start,
                                                     self._time_step)

        self._current_vel = sub_current_vel[-1]
        self._current_pos = sub_current_pos[-1]

        self._sub_acc.extend(list(normalize_batch(sub_current_acc, self._acc_limits_min_max)))
        self._sub_vel.extend(list(normalize_batch(sub_current_vel, self._vel_limits_min_max)))
        self._sub_pos.extend(list(normalize_batch(sub_current_pos, self._pos_limits_min_max)))

        self._acc.append(self._sub_acc[-1])
        self._vel.append(self._sub_vel[-1])
        self._pos.append(self._sub_pos[-1])

        self._time_step_counter = self._time_step_counter + 1

    def get_trajectory_summary(self):
        joint_sub_data = {'pos': np.swapaxes(self._sub_pos, 0, 1),
                          'vel': np.swapaxes(self._sub_vel, 0, 1),
                          'acc': np.swapaxes(self._sub_acc, 0, 1),
                          'jerk': np.swapaxes(self._sub_jerk, 0, 1),
                          'violation_code': np.swapaxes(self._violation, 0, 1)}

        trajectory_summary = {}

        for key, value in joint_sub_data.items():
            trajectory_summary[key] = []
            for joint_data in value:
                trajectory_summary[key].append({'min': np.min(joint_data),
                                                'max': np.max(joint_data),
                                                'final': joint_data[-1]})

        return trajectory_summary


def normalize_slow(value, value_range):
    normalized_value = -1 + 2 * (value - value_range[0]) / (value_range[1] - value_range[0])
    return normalized_value


def denormalize_slow(norm_value, value_range):
    actual_value = value_range[0] + 0.5 * (norm_value + 1) * (value_range[1] - value_range[0])
    return actual_value
