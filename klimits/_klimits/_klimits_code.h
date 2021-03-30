#ifndef KLIMITS_CODE_H
#define KLIMITS_CODE_H

#include <complex.h>

double pos_all_a1_max(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min);
double pos_all_a1_min(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min);
double pos_all_bounded_vel_continuous_a1(double j_min, double j_max, double a_0, double a_min, double v_0, double t_s, double t_star, double t_n_a_min, double t_u);
double complex pos_all_bounded_vel_continuous_tu(double j_min, double j_max, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_star, double t_n_a_min);
double pos_all_bounded_vel_discrete_a1(double j_min, double j_max, double j_n_u_plus_1, double a_0, double a_min, double v_0, double t_s, double t_star, double t_n_a_min, double t_n_u);
double pos_all_bounded_vel_discrete_j_n_u_plus_1(double j_min, double j_max, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_star, double t_n_a_min, double t_n_u);
double pos_all_tv0(double j_min, double a_0, double a_1, double a_min, double v_0, double t_s, double t_n_a_min);
double pos_first_a1(double a_0, double v_0, double t_s, double t_v0);
double pos_first_tv0_max(double a_0, double v_0, double p_0, double p_max);
double pos_first_tv0_min(double a_0, double v_0, double p_0, double p_max);
double pos_min_jerk_a1(double j_min, double a_0, double v_0, double t_s, double t_v0);
double pos_min_jerk_bounded_vel_continuous_a1(double j_min, double j_max, double a_0, double v_0, double t_s, double t_star, double t_u);
double complex pos_min_jerk_bounded_vel_continuous_tu(double j_min, double j_max, double a_0, double v_0, double p_0, double p_max, double t_s, double t_star);
double pos_min_jerk_bounded_vel_discrete_a1(double j_min, double j_max, double j_n_u_plus_1, double a_0, double v_0, double t_s, double t_star, double t_n_u);
double pos_min_jerk_bounded_vel_discrete_j_n_u_plus_1(double j_min, double j_max, double a_0, double v_0, double p_0, double p_max, double t_s, double t_star, double t_n_u);
double pos_min_jerk_tv0_0(double j_min, double a_0, double v_0, double p_0, double p_max, double t_s);
double pos_min_jerk_tv0_1(double j_min, double a_0, double v_0, double p_0, double p_max, double t_s);
double complex pos_min_jerk_tv0_2(double j_min, double a_0, double v_0, double p_0, double p_max, double t_s);
double pos_reduced_jerk_a1(double j_min, double a_0, double a_min, double v_0, double t_s, double t_v0, double t_n_a_min);
double pos_reduced_jerk_tv0_0(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min);
double pos_reduced_jerk_tv0_1(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min);
double complex pos_reduced_jerk_tv0_2(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min);
double pos_reduced_jerk_tv0_3(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min);
double pos_reduced_jerk_tv0_4(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min);
double complex pos_reduced_jerk_tv0_5(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min);
double pos_upper_bound_a1_max_0(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s);
double pos_upper_bound_a1_max_1(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s);
double complex pos_upper_bound_a1_max_2(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s);
double pos_upper_bound_a1_max_3(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s);
double pos_upper_bound_a1_max_4(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s);
double complex pos_upper_bound_a1_max_5(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s);
double pos_upper_bound_a1_min_0(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s);
double pos_upper_bound_a1_min_1(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s);
double complex pos_upper_bound_a1_min_2(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s);
double pos_upper_bound_a1_min_3(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s);
double pos_upper_bound_a1_min_4(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s);
double complex pos_upper_bound_a1_min_5(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s);
double pos_upper_bound_tv0(double j_min, double a_0, double a_1, double a_min, double v_0, double t_s);
double vel_reduced_a1_max(double j_min, double a_0, double a_n_plus_1_star, double v_0, double v_max, double t_s, double t_n);
double vel_reduced_a1_min(double j_min, double a_0, double a_n_plus_1_star, double v_0, double v_max, double t_s, double t_n);

#endif

