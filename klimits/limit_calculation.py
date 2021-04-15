# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import math
from multiprocessing import Pool
import numpy as np
import os
import sys
import inspect
sys.path.append(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
from joint_limit_equations import JointLimitEquations


class PosVelJerkLimitation:
    def __init__(self,
                 time_step,
                 pos_limits,
                 vel_limits,
                 acc_limits,
                 jerk_limits,
                 acceleration_after_max_vel_limit_factor=0.0001,
                 set_velocity_after_max_pos_to_zero=True,
                 limit_velocity=True,
                 limit_position=True,
                 num_workers=1,
                 soft_velocity_limits=False,
                 soft_position_limits=False,
                 *vargs,
                 **kwargs):

        self._time_step = time_step
        self._pos_limits = pos_limits
        self._vel_limits = vel_limits
        self._num_joints = len(self._vel_limits)
        self._acc_limits = acc_limits
        self._jerk_limits = jerk_limits

        self._acceleration_after_max_vel_limit_factor = acceleration_after_max_vel_limit_factor
        self._set_velocity_after_max_pos_to_zero = set_velocity_after_max_pos_to_zero
        self._limit_velocity = limit_velocity
        self._limit_position = limit_position
        self._soft_velocity_limits = soft_velocity_limits
        self._soft_position_limits = soft_position_limits

        self._worker_pool = None
        self._joint_limit_equations = JointLimitEquations()

        if num_workers >= 2 and self._num_joints >= 2:
            self._worker_pool = Pool(min(num_workers, self._num_joints))
        else:
            self._worker_pool = None

    @property
    def set_velocity_after_max_pos_to_zero(self):
        return self._set_velocity_after_max_pos_to_zero

    @set_velocity_after_max_pos_to_zero.setter
    def set_velocity_after_max_pos_to_zero(self, set_velocity_after_max_pos_to_zero):
        self._set_velocity_after_max_pos_to_zero = set_velocity_after_max_pos_to_zero

    @property
    def pos_limits(self):
        return self._pos_limits

    @pos_limits.setter
    def pos_limits(self, pos_limits):
        self._pos_limits = pos_limits

    @property
    def vel_limits(self):
        return self._vel_limits

    @vel_limits.setter
    def vel_limits(self, vel_limits):
        self._vel_limits = vel_limits

    @property
    def acc_limits(self):
        return self._acc_limits

    @acc_limits.setter
    def acc_limits(self, acc_limits):
        self._acc_limits = acc_limits

    def calculate_valid_acceleration_range(self, current_pos, current_vel, current_acc, braking_trajectory=False,
                                           time_step_counter=0):

        if self._worker_pool:
            pool_result = self._worker_pool.starmap(self._calculate_valid_acceleration_range_per_joint,
                                                    [(
                                                     i, self._time_step, current_pos[i], current_vel[i], current_acc[i],
                                                     self._pos_limits[i], self._vel_limits[i], self._acc_limits[i],
                                                     self._jerk_limits[i],
                                                     self._acceleration_after_max_vel_limit_factor,
                                                     self._set_velocity_after_max_pos_to_zero,
                                                     self._limit_velocity, self._limit_position, braking_trajectory,
                                                     time_step_counter)
                                                     for i in range(self._num_joints)])

            pool_result = np.swapaxes(pool_result, 0, 1)
            norm_acc_range = pool_result[0]
            limit_violation = pool_result[1]

        else:
            norm_acc_range = []
            limit_violation = []

            for i in range(self._num_joints):
                pos = current_pos[i] if current_pos is not None else None
                pos_limits = self._pos_limits[i] if self._pos_limits is not None else None
                norm_acc_range_joint, limit_violation_joint = \
                    self._calculate_valid_acceleration_range_per_joint(i, self._time_step, pos, current_vel[i],
                                                                       current_acc[i], pos_limits, self._vel_limits[i],
                                                                       self._acc_limits[i], self._jerk_limits[i],
                                                                       self._acceleration_after_max_vel_limit_factor,
                                                                       self._set_velocity_after_max_pos_to_zero,
                                                                       self._limit_velocity, self._limit_position,
                                                                       braking_trajectory,
                                                                       time_step_counter)

                norm_acc_range.append(norm_acc_range_joint)
                limit_violation.append(limit_violation_joint)

        return norm_acc_range, limit_violation

    def _calculate_valid_acceleration_range_per_joint(self, joint_index, t_s, current_pos, current_vel, current_acc,
                                                      pos_limits, vel_limits, acc_limits, jerk_limits,
                                                      acceleration_after_max_vel_limit_factor,
                                                      set_velocity_after_max_pos_to_zero=False,
                                                      limit_velocity=True, limit_position=True,
                                                      braking_trajectory=False, time_step_counter=0):

        acc_range_jerk = [current_acc + jerk_limits[0] * t_s,
                          current_acc + jerk_limits[1] * t_s]
        acc_range_acc = [acc_limits[0], acc_limits[1]]

        acc_range_dynamic_vel = [acc_limits[0], acc_limits[1]]

        if limit_velocity:
            if (current_acc < 0 and (
                    current_vel < vel_limits[0] + 0.5 * (current_acc ** 2 * t_s) / (acc_limits[1] - current_acc))) \
                    and not braking_trajectory:
                acc_range_dynamic_vel = [acc_limits[1], acc_limits[1]]
            else:
                if (current_acc > 0 and (
                        current_vel > vel_limits[1] - 0.5 * (current_acc ** 2 * t_s) / (
                        current_acc - acc_limits[0]))) and not braking_trajectory:
                    acc_range_dynamic_vel = [acc_limits[0], acc_limits[0]]
                else:
                    for j in range(2):
                        nj = (j + 1) % 2
                        if (j == 0 and (current_vel + 0.5 * current_acc * t_s) <= vel_limits[0]) \
                                or (j == 1 and (current_vel + 0.5 * current_acc * t_s) >= vel_limits[1]):

                            if current_acc == 0:
                                acc_range_dynamic_vel[j] = 0
                            else:
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    acc_range_dynamic_vel[j] = current_acc * (
                                            1 - ((0.5 * current_acc * t_s) / (vel_limits[j] - current_vel)))

                        else:
                            a = - jerk_limits[nj] / 2
                            b = t_s * jerk_limits[nj] / 2
                            c = current_vel - vel_limits[j] + current_acc * t_s / 2

                            if b ** 2 - 4 * a * c >= 0:
                                if j == 0:
                                    t_a0_1 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (
                                            2 * a)
                                else:
                                    t_a0_1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (
                                            2 * a)

                                a1_limit = - jerk_limits[nj] * (t_a0_1 - t_s)

                                if np.ceil(t_a0_1 / t_s) > t_a0_1 / t_s:
                                    n = np.ceil(t_a0_1 / t_s) - 1
                                    a_n_plus_1 = a1_limit + jerk_limits[nj] * t_s * n
                                    if (j == 0 and a_n_plus_1 > acc_limits[1] *
                                        acceleration_after_max_vel_limit_factor) or \
                                            (j == 1 and a_n_plus_1 < acc_limits[0] *
                                             acceleration_after_max_vel_limit_factor):
                                        a_0 = current_acc
                                        a_n_plus_1_star = acc_limits[nj] * acceleration_after_max_vel_limit_factor
                                        t_n = n * t_s
                                        j_min = jerk_limits[nj]
                                        v_0 = current_vel
                                        v_max = vel_limits[j]
                                        a1_limit = \
                                            self._joint_limit_equations.velocity_reduced_acceleration(j, j_min,
                                                                                                      a_0,
                                                                                                      a_n_plus_1_star,
                                                                                                      v_0, v_max,
                                                                                                      t_s, t_n)

                                acc_range_dynamic_vel[j] = a1_limit
                            else:
                                pass

        acc_range_dynamic_pos = [-10 ** 6, 10 ** 6]

        if limit_position:
            for j in range(2):
                nj = (j + 1) % 2
                a_min = acc_limits[nj]
                j_min = jerk_limits[nj]
                j_max = jerk_limits[j]
                p_max = pos_limits[j]
                p_0 = current_pos
                v_0 = current_vel
                a_0 = current_acc
                a_1_all_phases = 0
                a_1_reduced_jerk = None
                t_v0_bounded_vel_min_jerk_phase = None
                t_star_all_phases = None

                a_1_min_jerk, t_v0_min_jerk = self._joint_limit_equations.position_border_case_min_jerk_phase(j_min,
                                                                                                              a_0, v_0,
                                                                                                              p_0,
                                                                                                              p_max,
                                                                                                              t_s)

                if t_v0_min_jerk < t_s + 1e-8 or math.isnan(t_v0_min_jerk):
                    a_1_min_first, t_v0_min_first = self._joint_limit_equations.position_border_case_first_phase(j, a_0,
                                                                                                                 v_0,
                                                                                                                 p_0,
                                                                                                                 p_max,
                                                                                                                 t_s)

                    if 0 < t_v0_min_first <= t_s + 1e-3:
                        acc_range_dynamic_pos[j] = a_1_min_first
                    else:

                        if p_0 == p_max and a_0 == 0 and v_0 == 0:
                            acc_range_dynamic_pos[j] = 0
                else:

                    t_n_a_min = t_s * (1 + np.floor((a_min - a_1_min_jerk) / (j_min * t_s)))

                    if t_n_a_min >= t_v0_min_jerk:
                        if t_v0_min_jerk >= t_s:
                            acc_range_dynamic_pos[j] = a_1_min_jerk

                            if set_velocity_after_max_pos_to_zero:
                                t_v0_bounded_vel_min_jerk_phase = t_v0_min_jerk

                    else:
                        t_a_min = t_s * (1 + ((a_min - a_1_min_jerk) / (j_min * t_s)))

                        if t_v0_min_jerk > t_a_min:
                            a_1_upper_bound, t_v0_upper_bound = \
                                self._joint_limit_equations.position_border_case_upper_bound(j, j_min, a_0, a_min, v_0,
                                                                                             p_0, p_max, t_s)
                            if math.isnan(a_1_upper_bound):
                                acc_range_dynamic_pos[j] = a_0 + j_min
                                continue

                            t_a_min_upper_bound = t_s * (1 + ((a_min - a_1_upper_bound) / (j_min * t_s)))
                            if t_a_min_upper_bound < t_s:
                                if t_a_min_upper_bound / t_s > 0.999:
                                    t_a_min_upper_bound = t_s

                            t_n_a_min = t_s * np.floor(t_a_min_upper_bound / t_s)

                        a_1_all_phases, t_v0_all_phases = \
                            self._joint_limit_equations.position_border_case_all_phases(j, j_min, a_0, a_min, v_0, p_0,
                                                                                        p_max, t_s, t_n_a_min)

                        if t_v0_all_phases >= t_n_a_min + t_s:
                            acc_range_dynamic_pos[j] = a_1_all_phases

                            if set_velocity_after_max_pos_to_zero:
                                t_star_all_phases = t_s * np.ceil(t_v0_all_phases / t_s)
                                a_1_bounded_vel_continuous_all_phases, t_u_bounded_vel_continuous_all_phases = \
                                    self._joint_limit_equations. \
                                        position_bounded_velocity_continuous_all_phases(j_min, j_max, a_0, a_min, v_0,
                                                                                        p_0, p_max, t_s,
                                                                                        t_star_all_phases, t_n_a_min)

                                if t_u_bounded_vel_continuous_all_phases >= t_star_all_phases:
                                    pass

                                if t_u_bounded_vel_continuous_all_phases >= t_n_a_min + t_s:
                                    t_n_u_all_phases = t_s * np.floor(t_u_bounded_vel_continuous_all_phases / t_s)
                                    a_1_bounded_vel_discrete_all_phases, j_n_u_plus_1_all_phases = \
                                        self._joint_limit_equations. \
                                            position_bounded_velocity_discrete_all_phases(j_min, j_max, a_0, a_min, v_0,
                                                                                          p_0, p_max, t_s,
                                                                                          t_star_all_phases, t_n_a_min,
                                                                                          t_n_u_all_phases)

                                    a_n_a_min = a_1_bounded_vel_discrete_all_phases + (t_n_a_min - t_s) * j_min
                                    if (j == 0 and a_n_a_min > a_min + 1e-3) or \
                                            (j == 1 and a_n_a_min < a_min - 1e-3):

                                        if round(t_n_a_min / t_s) > 1:
                                            a_1_bounded_vel_discrete_all_phases, j_n_u_plus_1_all_phases = \
                                                self._joint_limit_equations. \
                                                    position_bounded_velocity_discrete_all_phases(j_min, j_max, a_0,
                                                                                                  a_min, v_0, p_0,
                                                                                                  p_max, t_s,
                                                                                                  t_star_all_phases,
                                                                                                  t_n_a_min - t_s,
                                                                                                  t_n_u_all_phases)

                                            a_n_a_min = a_1_bounded_vel_discrete_all_phases + (
                                                    t_n_a_min - 2 * t_s) * j_min

                                    if (j == 0 and a_n_a_min > a_min + 1e-3) or \
                                            (j == 1 and a_n_a_min < a_min - 1e-3):
                                        pass
                                    else:
                                        if (j == 0 and j_max <= j_n_u_plus_1_all_phases <= 0) or \
                                                (j == 1 and 0 <= j_n_u_plus_1_all_phases <= j_max):

                                            if (j == 0 and a_1_bounded_vel_discrete_all_phases > a_1_all_phases) or \
                                                    (j == 1 and a_1_bounded_vel_discrete_all_phases < a_1_all_phases):

                                                acc_range_dynamic_pos[j] = a_1_bounded_vel_discrete_all_phases
                                            else:
                                                pass
                                        else:
                                            pass
                                else:
                                    t_v0_bounded_vel_min_jerk_phase = t_v0_all_phases
                        else:
                            if set_velocity_after_max_pos_to_zero:
                                t_v0_bounded_vel_min_jerk_phase = t_v0_all_phases

                            a_1_reduced_jerk, t_v0_reduced_jerk = self._joint_limit_equations. \
                                position_border_case_reduced_jerk_phase(j_min, a_0, a_min, v_0, p_0, p_max, t_s,
                                                                        t_n_a_min)

                            acc_range_dynamic_pos[j] = a_1_reduced_jerk

                    if set_velocity_after_max_pos_to_zero and t_v0_bounded_vel_min_jerk_phase:
                        t_star_min_jerk_phase = t_s * np.ceil(
                            t_v0_bounded_vel_min_jerk_phase / t_s)

                        if t_star_min_jerk_phase >= 3 * t_s:
                            a_1_bounded_vel_continuous_min_jerk, t_u_bounded_vel_continuous_min_jerk = \
                                self._joint_limit_equations.\
                                    position_bounded_velocity_continuous_min_jerk_phase(j_min, j_max, a_0, v_0, p_0,
                                                                                        p_max, t_s,
                                                                                        t_star_min_jerk_phase)

                            if not math.isnan(t_u_bounded_vel_continuous_min_jerk) and \
                                    (t_u_bounded_vel_continuous_min_jerk / t_s) > 0.99:
                                t_n_u_min_jerk_phase = max(t_s * np.floor(t_u_bounded_vel_continuous_min_jerk / t_s),
                                                           t_s)
                            else:
                                t_n_u_min_jerk_phase = float('nan')

                        else:
                            t_n_u_min_jerk_phase = t_s  #

                        if not math.isnan(t_n_u_min_jerk_phase):
                            a_1_bounded_vel_discrete_min_jerk, j_n_u_plus_1_min_jerk_phase = \
                                self._joint_limit_equations. \
                                    position_bounded_velocity_discrete_min_jerk_phase(j_min, j_max, a_0, v_0, p_0,
                                                                                      p_max, t_s,
                                                                                      t_star_min_jerk_phase,
                                                                                      t_n_u_min_jerk_phase)
                            if (j == 0 and j_max - 1e-6 <= j_n_u_plus_1_min_jerk_phase <= j_min + 1e-6) or \
                                    (j == 1 and j_min - 1e-6 <= j_n_u_plus_1_min_jerk_phase <= j_max + 1e-6):

                                if t_star_all_phases is not None:
                                    if (j == 0 and a_1_bounded_vel_discrete_min_jerk > a_1_all_phases) or \
                                            (j == 1 and a_1_bounded_vel_discrete_min_jerk < a_1_all_phases):

                                        acc_range_dynamic_pos[j] = a_1_bounded_vel_discrete_min_jerk
                                    else:
                                        pass
                                else:
                                    if a_1_reduced_jerk is not None:
                                        if (j == 0 and a_1_bounded_vel_discrete_min_jerk > a_1_reduced_jerk) or \
                                                (j == 1 and a_1_bounded_vel_discrete_min_jerk < a_1_reduced_jerk):
                                            acc_range_dynamic_pos[j] = a_1_bounded_vel_discrete_min_jerk
                                        else:
                                            pass
                                    else:
                                        if (j == 0 and a_1_bounded_vel_discrete_min_jerk > a_1_min_jerk) or \
                                                (j == 1 and a_1_bounded_vel_discrete_min_jerk < a_1_min_jerk):
                                            acc_range_dynamic_pos[j] = a_1_bounded_vel_discrete_min_jerk
                                        else:
                                            pass
                            else:
                                pass
                        else:
                            pass

                if math.isnan(acc_range_dynamic_pos[j]):
                    acc_range_dynamic_pos[j] = acc_limits[j]

        acc_range_list = []
        acc_range_list.append(acc_range_jerk)
        acc_range_list.append(acc_range_acc)

        if limit_velocity:
            if self._soft_velocity_limits:
                acc_range_dynamic_vel[0] = max(acc_range_jerk[0], acc_range_dynamic_vel[0])
                acc_range_dynamic_vel[1] = min(acc_range_jerk[1], acc_range_dynamic_vel[1])

                if acc_range_dynamic_vel[1] < acc_range_jerk[0]:
                    acc_range_dynamic_vel[1] = acc_range_jerk[0]
                if acc_range_dynamic_vel[0] > acc_range_jerk[1]:
                    acc_range_dynamic_vel[0] = acc_range_jerk[1]

            acc_range_list.append(acc_range_dynamic_vel)

        if limit_position:
            if self._soft_position_limits and self._soft_velocity_limits:
                acc_range_dynamic_pos[0] = max(acc_range_jerk[0], acc_range_dynamic_pos[0])
                acc_range_dynamic_pos[1] = min(acc_range_jerk[1], acc_range_dynamic_pos[1])

                if acc_range_dynamic_pos[1] < acc_range_jerk[0]:
                    acc_range_dynamic_pos[1] = acc_range_jerk[0]
                if acc_range_dynamic_pos[0] > acc_range_jerk[1]:
                    acc_range_dynamic_pos[0] = acc_range_jerk[1]

            acc_range_list.append(acc_range_dynamic_pos)

        limit_violation_code = 0

        for j in range(len(acc_range_list)):
            if j <= limit_violation_code:
                acc_range_swap = np.swapaxes(acc_range_list, 0, 1)
                acc_range_total = [np.max(acc_range_swap[0][j:]), np.min(acc_range_swap[1][j:])]
                if math.isnan(acc_range_total[0]) or math.isnan(acc_range_total[1]) or \
                        (acc_range_total[0] - acc_range_total[1]) > 0.001:
                    limit_violation_code = limit_violation_code + 1
                else:
                    if (acc_range_total[0] - acc_range_total[1]) > 0:
                        acc_range_total[1] = acc_range_total[0]

        norm_acc_range_joint = [normalize(accLimit, acc_limits) for accLimit in acc_range_total]
        norm_acc_range_joint = list(np.clip(norm_acc_range_joint, -1, 1))

        return norm_acc_range_joint, limit_violation_code


def normalize(value, value_range):
    normalized_value = -1 + 2 * (value - value_range[0]) / (value_range[1] - value_range[0])
    return normalized_value


def denormalize(norm_value, value_range):
    actual_value = value_range[0] + 0.5 * (norm_value + 1) * (value_range[1] - value_range[0])
    return actual_value


