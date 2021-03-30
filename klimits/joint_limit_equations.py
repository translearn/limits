# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import os
import sys
import inspect
sys.path.append(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
import _klimits


class JointLimitEquations:
    def __init__(self):
        self._a_1_velocity_reduced_acceleration = []

        self._a_1_velocity_reduced_acceleration.append(
            load_expression(phase_and_variable="vel_reduced_a1", min_max=0, expression_number=-1))

        self._a_1_velocity_reduced_acceleration.append(
            load_expression(phase_and_variable="vel_reduced_a1", min_max=1, expression_number=-1))

        self._a_1_position_border_case_min_jerk_phase = load_expression(phase_and_variable="pos_min_jerk_a1",
                                                                        min_max=-1, expression_number=-1)

        self._t_v0_position_border_case_min_jerk_phase = []

        for i in range(3):
            self._t_v0_position_border_case_min_jerk_phase.append(
                load_expression(phase_and_variable="pos_min_jerk_tv0",
                                min_max=-1, expression_number=i))

        self._a_1_position_border_case_first_phase = load_expression(phase_and_variable="pos_first_a1",
                                                                     min_max=-1, expression_number=-1)

        self._t_v0_position_border_case_first_phase = []
        self._a_1_position_border_case_upper_bound = [[], []]

        for min_max in range(2):
            self._t_v0_position_border_case_first_phase.append(load_expression(phase_and_variable="pos_first_tv0",
                                                                               min_max=min_max, expression_number=-1))
            for solution in range(2):
                for i in range(3):
                    self._a_1_position_border_case_upper_bound[min_max].append(
                        load_expression(phase_and_variable="pos_upper_bound_a1",
                                        min_max=min_max, expression_number=i + solution * 3))

        self._t_v0_position_border_case_upper_bound = load_expression(phase_and_variable="pos_upper_bound_tv0",
                                                                      min_max=-1, expression_number=-1)

        self._a_1_position_border_case_all_phases = []

        for min_max in range(2):
            self._a_1_position_border_case_all_phases.append(load_expression(phase_and_variable="pos_all_a1",
                                                                             min_max=min_max, expression_number=-1))

        self._t_v0_position_border_case_all_phases = load_expression(phase_and_variable="pos_all_tv0", min_max=-1,
                                                                     expression_number=-1)

        self._a_1_position_border_case_reduced_jerk_phase = load_expression(phase_and_variable="pos_reduced_jerk_a1",
                                                                            min_max=-1, expression_number=-1)

        self._t_v0_position_border_case_reduced_jerk_phase = []

        for solution in range(2):
            for i in range(3):
                self._t_v0_position_border_case_reduced_jerk_phase.append(
                    load_expression(phase_and_variable="pos_reduced_jerk_tv0", min_max=-1,
                                    expression_number=i + solution * 3))

        self._a_1_position_bounded_velocity_continuous_all_phases = load_expression(
            phase_and_variable="pos_all_bounded_vel_continuous_a1", min_max=-1, expression_number=-1)

        self._t_u_position_bounded_velocity_continuous_all_phases = load_expression(
            phase_and_variable="pos_all_bounded_vel_continuous_tu", min_max=-1, expression_number=-1)

        self._a_1_position_bounded_velocity_discrete_all_phases = load_expression(
            phase_and_variable="pos_all_bounded_vel_discrete_a1", min_max=-1, expression_number=-1)

        self._j_n_u_plus_1_position_bounded_velocity_discrete_all_phases = load_expression(
            phase_and_variable="pos_all_bounded_vel_discrete_j_n_u_plus_1", min_max=-1, expression_number=-1)

        self._a_1_position_bounded_velocity_continuous_min_jerk_phase = load_expression(
            phase_and_variable="pos_min_jerk_bounded_vel_continuous_a1", min_max=-1, expression_number=-1)

        self._t_u_position_bounded_velocity_continuous_min_jerk_phase = load_expression(
            phase_and_variable="pos_min_jerk_bounded_vel_continuous_tu", min_max=-1, expression_number=-1)

        self._a_1_position_bounded_velocity_discrete_min_jerk_phase = load_expression(
            phase_and_variable="pos_min_jerk_bounded_vel_discrete_a1", min_max=-1, expression_number=-1)

        self._j_n_u_plus_1_position_bounded_velocity_discrete_min_jerk_phase = load_expression(
            phase_and_variable="pos_min_jerk_bounded_vel_discrete_j_n_u_plus_1", min_max=-1, expression_number=-1)

    def velocity_reduced_acceleration(self, min_max, j_min_in, a_0_in, a_n_plus_1_star_in, v_0_in, v_max_in, t_s_in,
                                      t_n_in):

        a_1_out = self._a_1_velocity_reduced_acceleration[min_max](j_min_in, a_0_in, a_n_plus_1_star_in, v_0_in,
                                                                   v_max_in, t_s_in, t_n_in)
        return a_1_out
    
    def position_border_case_min_jerk_phase(self, j_min_in, a_0_in, v_0_in, p_0_in, p_max_in, t_s_in):

        if self._t_v0_position_border_case_min_jerk_phase[0](j_min_in, a_0_in, v_0_in, p_0_in, p_max_in, t_s_in) == 0:
            t_v0_out = self._t_v0_position_border_case_min_jerk_phase[1](j_min_in, a_0_in, v_0_in, p_0_in, p_max_in,
                                                                         t_s_in)
        else:
            t_v0_out = self._t_v0_position_border_case_min_jerk_phase[2](j_min_in, a_0_in, v_0_in, p_0_in, p_max_in,
                                                                         t_s_in)

        if np.abs(np.imag(t_v0_out)) < 1e-5:
            t_v0_out = np.real(t_v0_out)
        else:
            t_v0_out = float('nan')

        a_1_out = self._a_1_position_border_case_min_jerk_phase(j_min_in, a_0_in, v_0_in, t_s_in, t_v0_out)

        return a_1_out, t_v0_out

    def position_border_case_first_phase(self, min_max, a_0_in, v_0_in, p_0_in, p_max_in, t_s_in):

        t_v0_out = self._t_v0_position_border_case_first_phase[min_max](a_0_in, v_0_in, p_0_in, p_max_in)
        a_1_out = self._a_1_position_border_case_first_phase(a_0_in, v_0_in, t_s_in, t_v0_out)

        return a_1_out, t_v0_out

    def position_border_case_upper_bound(self, min_max, j_min_in, a_0_in, a_min_in, v_0_in, p_0_in, p_max_in, t_s_in):

        for solution in range(2):
            a_1_position_border_case_upper_bound_condition = \
                self._a_1_position_border_case_upper_bound[min_max][0 + solution * 3](j_min_in, a_0_in, a_min_in,
                                                                                      v_0_in,
                                                                                      p_0_in, p_max_in, t_s_in)

            if a_1_position_border_case_upper_bound_condition == 0:
                a_1_out = self._a_1_position_border_case_upper_bound[min_max][1 + solution * 3](j_min_in, a_0_in,
                                                                                                a_min_in,
                                                                                                v_0_in, p_0_in,
                                                                                                p_max_in,
                                                                                                t_s_in)

            else:
                a_1_out = self._a_1_position_border_case_upper_bound[min_max][2 + solution * 3](j_min_in, a_0_in, a_min_in,
                                                                                          v_0_in, p_0_in, p_max_in,
                                                                                          t_s_in)

            if np.abs(np.imag(a_1_out)) < 1e-3:
                a_1_out = np.real(a_1_out)
                break

        if np.imag(a_1_out) != 0:
            a_1_out = float('nan')
            if min_max == 0:
                a_min_in = a_min_in - 0.02
            else:
                a_min_in = a_min_in + 0.02

            for solution in range(2):
                a_1_out = self._a_1_position_border_case_upper_bound[min_max][2 + solution * 3](j_min_in, a_0_in,
                                                                                                a_min_in, v_0_in,
                                                                                                p_0_in, p_max_in,
                                                                                                t_s_in)

                if np.abs(np.imag(a_1_out)) < 1e-3:
                    a_1_out = np.real(a_1_out)
                    break

            if np.imag(a_1_out) != 0:
                a_1_out = float('nan')

        t_v0_out = self._t_v0_position_border_case_upper_bound(j_min_in, a_0_in, a_1_out, a_min_in, v_0_in, t_s_in)

        return a_1_out, t_v0_out

    def position_border_case_all_phases(self, min_max, j_min_in, a_0_in, a_min_in, v_0_in, p_0_in, p_max_in, t_s_in,
                                        t_n_a_min_in):

        a_1_out = self._a_1_position_border_case_all_phases[min_max](j_min_in, a_0_in, a_min_in, v_0_in, p_0_in,
                                                                     p_max_in, t_s_in, t_n_a_min_in)
        t_v0_out = self._t_v0_position_border_case_all_phases(j_min_in, a_0_in, a_1_out, a_min_in, v_0_in, t_s_in,
                                                              t_n_a_min_in)

        return a_1_out, t_v0_out

    def position_border_case_reduced_jerk_phase(self, j_min_in, a_0_in, a_min_in, v_0_in, p_0_in, p_max_in, t_s_in,
                                                t_n_a_min_in):

        for solution in range(2):
            if self._t_v0_position_border_case_reduced_jerk_phase[0 + solution * 3](j_min_in, a_0_in, a_min_in, v_0_in,
                                                                                    p_0_in, p_max_in, t_s_in,
                                                                                    t_n_a_min_in) == 0:
                t_v0_out = self._t_v0_position_border_case_reduced_jerk_phase[1 + solution * 3](j_min_in, a_0_in,
                                                                                                a_min_in,
                                                                                                v_0_in, p_0_in,
                                                                                                p_max_in,
                                                                                                t_s_in,
                                                                                                t_n_a_min_in)
            else:
                t_v0_out = self._t_v0_position_border_case_reduced_jerk_phase[2 + solution * 3](j_min_in, a_0_in,
                                                                                                a_min_in,
                                                                                                v_0_in, p_0_in,
                                                                                                p_max_in,
                                                                                                t_s_in,
                                                                                                t_n_a_min_in)

            if np.abs(np.imag(t_v0_out)) < 1e-5 and np.real(t_v0_out) >= t_n_a_min_in:
                t_v0_out = np.real(t_v0_out)
                break

        if np.imag(t_v0_out) != 0:
            t_v0_out = float('nan')

        a_1_out = self._a_1_position_border_case_reduced_jerk_phase(j_min_in, a_0_in, a_min_in, v_0_in, t_s_in,
                                                                    t_v0_out,
                                                                    t_n_a_min_in)

        return a_1_out, t_v0_out

    def position_bounded_velocity_continuous_all_phases(self, j_min_in, j_max_in, a_0_in, a_min_in, v_0_in, p_0_in,
                                                        p_max_in, t_s_in, t_star_in, t_n_a_min_in):

        compute_a_1 = True
        t_u_out = self._t_u_position_bounded_velocity_continuous_all_phases(j_min_in, j_max_in, a_0_in, a_min_in,
                                                                            v_0_in,
                                                                            p_0_in, p_max_in, t_s_in, t_star_in,
                                                                            t_n_a_min_in)
        if np.abs(np.imag(t_u_out)) < 1e-8:
            t_u_out = np.real(t_u_out)
        else:
            t_u_out = float('nan')

        if compute_a_1:
            a_1_out = self._a_1_position_bounded_velocity_continuous_all_phases(j_min_in, j_max_in, a_0_in, a_min_in,
                                                                                v_0_in, t_s_in, t_star_in, t_n_a_min_in,
                                                                                t_u_out)
        else:
            a_1_out = float('nan')

        return a_1_out, t_u_out

    def position_bounded_velocity_discrete_all_phases(self, j_min_in, j_max_in, a_0_in, a_min_in, v_0_in, p_0_in,
                                                      p_max_in, t_s_in,
                                                      t_star_in, t_n_a_min_in, t_n_u_in):

        j_n_u_plus_1_out = self._j_n_u_plus_1_position_bounded_velocity_discrete_all_phases(j_min_in, j_max_in, a_0_in,
                                                                                            a_min_in, v_0_in, p_0_in,
                                                                                            p_max_in, t_s_in, t_star_in,
                                                                                            t_n_a_min_in, t_n_u_in)

        if np.abs(np.imag(j_n_u_plus_1_out)) < 1e-8:
            j_n_u_plus_1_out = np.real(j_n_u_plus_1_out)
        else:
            j_n_u_plus_1_out = float('nan')

        a_1_out = self._a_1_position_bounded_velocity_discrete_all_phases(j_min_in, j_max_in, j_n_u_plus_1_out, a_0_in,
                                                                          a_min_in, v_0_in, t_s_in, t_star_in,
                                                                          t_n_a_min_in, t_n_u_in)

        return a_1_out, j_n_u_plus_1_out

    def position_bounded_velocity_continuous_min_jerk_phase(self, j_min_in, j_max_in, a_0_in, v_0_in, p_0_in, p_max_in,
                                                            t_s_in, t_star_in):

        compute_a_1 = True
        t_u_out = self._t_u_position_bounded_velocity_continuous_min_jerk_phase(j_min_in, j_max_in, a_0_in, v_0_in,
                                                                                p_0_in, p_max_in, t_s_in, t_star_in)

        if np.abs(np.imag(t_u_out)) < 1e-8:
            t_u_out = np.real(t_u_out)
        else:
            t_u_out = float('nan')

        if compute_a_1:
            a_1_out = self._a_1_position_bounded_velocity_continuous_min_jerk_phase(j_min_in, j_max_in, a_0_in, v_0_in,
                                                                                    t_s_in,
                                                                                    t_star_in, t_u_out)
        else:
            a_1_out = float('nan')

        return a_1_out, t_u_out

    def position_bounded_velocity_discrete_min_jerk_phase(self, j_min_in, j_max_in, a_0_in, v_0_in, p_0_in, p_max_in,
                                                          t_s_in,
                                                          t_star_in, t_n_u_in):

        j_n_u_plus_1_out = self._j_n_u_plus_1_position_bounded_velocity_discrete_min_jerk_phase(j_min_in, j_max_in,
                                                                                                a_0_in, v_0_in, p_0_in,
                                                                                                p_max_in, t_s_in,
                                                                                                t_star_in, t_n_u_in)

        if np.abs(np.imag(j_n_u_plus_1_out)) < 1e-8:
            j_n_u_plus_1_out = np.real(j_n_u_plus_1_out)
        else:
            j_n_u_plus_1_out = float('nan')

        a_1_out = self._a_1_position_bounded_velocity_discrete_min_jerk_phase(j_min_in, j_max_in, j_n_u_plus_1_out,
                                                                              a_0_in, v_0_in,
                                                                              t_s_in, t_star_in, t_n_u_in)

        return a_1_out, j_n_u_plus_1_out


def load_expression(phase_and_variable, min_max=-1, expression_number=-1):
    c_function_name = get_c_function_name(phase_and_variable, min_max, expression_number)
    return getattr(_klimits, c_function_name)


def get_c_function_name(phase_and_variable, min_max=-1, expression_number=-1):

    if min_max == 0:
        min_max_str = "_min"
    else:
        if min_max == 1:
            min_max_str = "_max"
        else:
            min_max_str = ""

    if expression_number == -1:
        expression_str = ""
    else:
        expression_str = "_" + str(expression_number)

    c_function_name = phase_and_variable + min_max_str + expression_str + "_c"

    return c_function_name






