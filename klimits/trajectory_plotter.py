# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import datetime
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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
                 plot_time_limits=None):

        self._time_step = time_step
        self._control_time_step = control_time_step
        self._plot_safe_acc_limits = plot_safe_acc_limits
        self._plot_time_limits = plot_time_limits

        self._plot_num_sub_time_steps = int(1000 * time_step)
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

        self._sub_time = None
        self._sub_pos = None
        self._sub_vel = None
        self._sub_acc = None
        self._sub_jerk = None

        self._pos_limits = pos_limits
        self._vel_limits = vel_limits
        self._acc_limits = acc_limits
        self._jerk_limits = jerk_limits

        self._num_joints = len(self._pos_limits)

        if plot_joint is None:
            self._plotJoint = [True for _ in range(self._num_joints)]
        else:
            self._plotJoint = plot_joint

        self._episode_counter = 0
        self._zero_vector = [0 for _ in range(self._num_joints)]

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
        self._current_pos = initial_joint_position.copy()

        self._pos = []
        self._vel = []
        self._acc = []
        self._jerk = []
        self._current_acc_limits = []
        self._current_acc_limits.append([[0, 0] for _ in range(self._num_joints)])

        self._sub_pos = []
        self._sub_vel = []
        self._sub_acc = []
        self._sub_jerk = []

        self._time = [0]
        self._pos.append([normalize(self._current_pos[i], self._pos_limits[i]) for i in range(len(self._current_pos))])
        self._vel.append([normalize(self._current_vel[i], self._vel_limits[i]) for i in range(len(self._current_vel))])
        self._acc.append([normalize(self._current_acc[i], self._acc_limits[i]) for i in range(len(self._current_acc))])
        self._jerk.append(self._zero_vector.copy())  

        self._sub_time = [0]
        self._sub_pos.append(self._pos[0].copy())
        self._sub_vel.append(self._vel[0].copy())
        self._sub_acc.append(self._acc[0].copy())
        self._sub_jerk.append(self._jerk[0].copy())

    def display_plot(self, max_time=None):

        num_subplots = 4
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

        joint_pos = np.swapaxes(self._pos, 0, 1)
        joint_vel = np.swapaxes(self._vel, 0, 1)
        joint_acc = np.swapaxes(self._acc, 0, 1)
        joint_jerk = np.swapaxes(self._jerk, 0, 1)

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

            if self._plotJoint[j]:
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

        for i in range(len(ax)):
            ax[i].legend(loc='lower right')
            if self._plot_time_limits is None:
                ax[i].set_xlim([0, self._time[-1]])  
            else:
                ax[i].set_xlim([0, self._plot_time_limits[1] - self._plot_time_limits[0]])

        if ax_acc is not None:
            ax[ax_acc].set_ylim([-1.05, 1.05])
        if ax_jerk is not None:
            ax[ax_jerk].set_ylim([-1.05, 1.05])

        fig.align_ylabels(ax)
        fig.set_size_inches((24.1, 13.5), forward=False)
        plt.show()

    def add_data_point(self, normalized_acc, normalized_acc_range=None):
        self._time.append(self._time[-1] + self._time_step)
        last_acc = self._current_acc.copy()
        last_vel = self._current_vel.copy()
        last_pos = self._current_pos.copy()
        self._current_acc = [denormalize(normalized_acc[k], self._acc_limits[k]) for k in
                             range(len(normalized_acc))]
        self._current_jerk = [(self._current_acc[k] - last_acc[k]) / self._time_step
                              for k in range(len(self._current_acc))]
        self._current_vel = [last_vel[k] + 0.5 * self._time_step * (last_acc[k] + self._current_acc[k])
                             for k in range(len(self._current_vel))]
        self._current_pos = [self._current_pos[k] + last_vel[k] * self._time_step
                             + (1 / 3 * last_acc[k] + 1 / 6 * self._current_acc[k]) * self._time_step ** 2
                             for k in range(len(self._current_pos))]

        self._pos.append([normalize(self._current_pos[k], self._pos_limits[k])
                          for k in range(len(self._current_pos))])

        self._vel.append([normalize(self._current_vel[k], self._vel_limits[k])
                          for k in range(len(self._current_vel))])

        self._jerk.append([normalize(self._current_jerk[k], self._jerk_limits[k])
                           for k in range(len(self._current_jerk))])

        self._acc.append(normalized_acc.tolist())
        self._current_acc_limits.append(normalized_acc_range)

        for j in range(1, self._plot_num_sub_time_steps + 1):
            t = j / self._plot_num_sub_time_steps * self._time_step
            self._sub_time.append(self._time_step_counter * self._time_step + t)
            self._sub_jerk.append(self._jerk[-1])
            sub_current_acc = [last_acc[k] + ((self._current_acc[k] - last_acc[k]) / self._time_step) * t
                               for k in range(len(self._current_acc))]
            sub_current_vel = [last_vel[k] + last_acc[k] * t +
                               0.5 * ((self._current_acc[k] - last_acc[k]) / self._time_step) * t ** 2
                               for k in range(len(self._current_vel))]
            sub_current_pos = [last_pos[k] + last_vel[k] * t + 0.5 * last_acc[k] * t ** 2 +
                               1 / 6 * ((self._current_acc[k] - last_acc[k]) / self._time_step) * t ** 3
                               for k in range(len(self._current_pos))]

            self._sub_acc.append([normalize(sub_current_acc[k], self._acc_limits[k])
                                  for k in range(len(sub_current_acc))])
            self._sub_vel.append([normalize(sub_current_vel[k], self._vel_limits[k])
                                  for k in range(len(sub_current_vel))])
            self._sub_pos.append([normalize(sub_current_pos[k], self._pos_limits[k])
                                  for k in range(len(sub_current_pos))])

        self._time_step_counter = self._time_step_counter + 1


def normalize(value, value_range):
    normalized_value = -1 + 2 * (value - value_range[0]) / (value_range[1] - value_range[0])
    return normalized_value


def denormalize(norm_value, value_range):
    actual_value = value_range[0] + 0.5 * (norm_value + 1) * (value_range[1] - value_range[0])
    return actual_value
