cdef extern from '_klimits_code.h':
    double pos_all_a1_max(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min)
    double pos_all_a1_min(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min)
    double pos_all_bounded_vel_continuous_a1(double j_min, double j_max, double a_0, double a_min, double v_0, double t_s, double t_star, double t_n_a_min, double t_u)
    double complex pos_all_bounded_vel_continuous_tu(double j_min, double j_max, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_star, double t_n_a_min)
    double pos_all_bounded_vel_discrete_a1(double j_min, double j_max, double j_n_u_plus_1, double a_0, double a_min, double v_0, double t_s, double t_star, double t_n_a_min, double t_n_u)
    double pos_all_bounded_vel_discrete_j_n_u_plus_1(double j_min, double j_max, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_star, double t_n_a_min, double t_n_u)
    double pos_all_tv0(double j_min, double a_0, double a_1, double a_min, double v_0, double t_s, double t_n_a_min)
    double pos_first_a1(double a_0, double v_0, double t_s, double t_v0)
    double pos_first_tv0_max(double a_0, double v_0, double p_0, double p_max)
    double pos_first_tv0_min(double a_0, double v_0, double p_0, double p_max)
    double pos_min_jerk_a1(double j_min, double a_0, double v_0, double t_s, double t_v0)
    double pos_min_jerk_bounded_vel_continuous_a1(double j_min, double j_max, double a_0, double v_0, double t_s, double t_star, double t_u)
    double complex pos_min_jerk_bounded_vel_continuous_tu(double j_min, double j_max, double a_0, double v_0, double p_0, double p_max, double t_s, double t_star)
    double pos_min_jerk_bounded_vel_discrete_a1(double j_min, double j_max, double j_n_u_plus_1, double a_0, double v_0, double t_s, double t_star, double t_n_u)
    double pos_min_jerk_bounded_vel_discrete_j_n_u_plus_1(double j_min, double j_max, double a_0, double v_0, double p_0, double p_max, double t_s, double t_star, double t_n_u)
    double pos_min_jerk_tv0_0(double j_min, double a_0, double v_0, double p_0, double p_max, double t_s)
    double pos_min_jerk_tv0_1(double j_min, double a_0, double v_0, double p_0, double p_max, double t_s)
    double complex pos_min_jerk_tv0_2(double j_min, double a_0, double v_0, double p_0, double p_max, double t_s)
    double pos_reduced_jerk_a1(double j_min, double a_0, double a_min, double v_0, double t_s, double t_v0, double t_n_a_min)
    double pos_reduced_jerk_tv0_0(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min)
    double pos_reduced_jerk_tv0_1(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min)
    double complex pos_reduced_jerk_tv0_2(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min)
    double pos_reduced_jerk_tv0_3(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min)
    double pos_reduced_jerk_tv0_4(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min)
    double complex pos_reduced_jerk_tv0_5(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min)
    double pos_upper_bound_a1_max_0(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s)
    double pos_upper_bound_a1_max_1(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s)
    double complex pos_upper_bound_a1_max_2(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s)
    double pos_upper_bound_a1_max_3(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s)
    double pos_upper_bound_a1_max_4(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s)
    double complex pos_upper_bound_a1_max_5(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s)
    double pos_upper_bound_a1_min_0(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s)
    double pos_upper_bound_a1_min_1(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s)
    double complex pos_upper_bound_a1_min_2(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s)
    double pos_upper_bound_a1_min_3(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s)
    double pos_upper_bound_a1_min_4(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s)
    double complex pos_upper_bound_a1_min_5(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s)
    double pos_upper_bound_tv0(double j_min, double a_0, double a_1, double a_min, double v_0, double t_s)
    double vel_reduced_a1_max(double j_min, double a_0, double a_n_plus_1_star, double v_0, double v_max, double t_s, double t_n)
    double vel_reduced_a1_min(double j_min, double a_0, double a_n_plus_1_star, double v_0, double v_max, double t_s, double t_n)


def pos_all_a1_max_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min):

    return pos_all_a1_max(j_min, a_0, a_min, v_0, p_0, p_max, t_s, t_n_a_min)


def pos_all_a1_min_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min):

    return pos_all_a1_min(j_min, a_0, a_min, v_0, p_0, p_max, t_s, t_n_a_min)
  

def pos_all_bounded_vel_continuous_a1_c(double j_min, double j_max, double a_0, double a_min, double v_0, double t_s, double t_star, double t_n_a_min, double t_u):

    return pos_all_bounded_vel_continuous_a1(j_min, j_max, a_0, a_min, v_0, t_s, t_star, t_n_a_min, t_u)


def pos_all_bounded_vel_continuous_tu_c(double j_min, double j_max, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_star, double t_n_a_min):

    return pos_all_bounded_vel_continuous_tu(j_min, j_max, a_0, a_min, v_0, p_0, p_max, t_s, t_star, t_n_a_min)


def pos_all_bounded_vel_discrete_a1_c(double j_min, double j_max, double j_n_u_plus_1, double a_0, double a_min, double v_0, double t_s, double t_star, double t_n_a_min, double t_n_u):

    return pos_all_bounded_vel_discrete_a1(j_min, j_max, j_n_u_plus_1, a_0, a_min, v_0, t_s, t_star, t_n_a_min, t_n_u)


def pos_all_bounded_vel_discrete_j_n_u_plus_1_c(double j_min, double j_max, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_star, double t_n_a_min, double t_n_u):

    return pos_all_bounded_vel_discrete_j_n_u_plus_1(j_min, j_max, a_0, a_min, v_0, p_0, p_max, t_s, t_star, t_n_a_min, t_n_u)


def pos_all_tv0_c(double j_min, double a_0, double a_1, double a_min, double v_0, double t_s, double t_n_a_min):

    return pos_all_tv0(j_min, a_0, a_1, a_min, v_0, t_s, t_n_a_min)
    

def pos_first_a1_c(double a_0, double v_0, double t_s, double t_v0):

    return pos_first_a1(a_0, v_0, t_s, t_v0)


def pos_first_tv0_max_c(double a_0, double v_0, double p_0, double p_max):

    return pos_first_tv0_max(a_0, v_0, p_0, p_max)
    

def pos_first_tv0_min_c(double a_0, double v_0, double p_0, double p_max):

    return pos_first_tv0_min(a_0, v_0, p_0, p_max)
  

def pos_min_jerk_a1_c(double j_min, double a_0, double v_0, double t_s, double t_v0):

    return pos_min_jerk_a1(j_min, a_0, v_0, t_s, t_v0)
    

def pos_min_jerk_bounded_vel_continuous_a1_c(double j_min, double j_max, double a_0, double v_0, double t_s, double t_star, double t_u):

    return pos_min_jerk_bounded_vel_continuous_a1(j_min, j_max, a_0, v_0, t_s, t_star, t_u)


def pos_min_jerk_bounded_vel_continuous_tu_c(double j_min, double j_max, double a_0, double v_0, double p_0, double p_max, double t_s, double t_star):

    return pos_min_jerk_bounded_vel_continuous_tu(j_min, j_max, a_0, v_0, p_0, p_max, t_s, t_star)


def pos_min_jerk_bounded_vel_discrete_a1_c(double j_min, double j_max, double j_n_u_plus_1, double a_0, double v_0, double t_s, double t_star, double t_n_u):

    return pos_min_jerk_bounded_vel_discrete_a1(j_min, j_max, j_n_u_plus_1, a_0, v_0, t_s, t_star, t_n_u)


def pos_min_jerk_bounded_vel_discrete_j_n_u_plus_1_c(double j_min, double j_max, double a_0, double v_0, double p_0, double p_max, double t_s, double t_star, double t_n_u):

    return pos_min_jerk_bounded_vel_discrete_j_n_u_plus_1(j_min, j_max, a_0, v_0, p_0, p_max, t_s, t_star, t_n_u)


def pos_min_jerk_tv0_0_c(double j_min, double a_0, double v_0, double p_0, double p_max, double t_s):

    return pos_min_jerk_tv0_0(j_min, a_0, v_0, p_0, p_max, t_s)


def pos_min_jerk_tv0_1_c(double j_min, double a_0, double v_0, double p_0, double p_max, double t_s):

    return pos_min_jerk_tv0_1(j_min, a_0, v_0, p_0, p_max, t_s)


def pos_min_jerk_tv0_2_c(double j_min, double a_0, double v_0, double p_0, double p_max, double t_s):

    return pos_min_jerk_tv0_2(j_min, a_0, v_0, p_0, p_max, t_s)


def pos_reduced_jerk_a1_c(double j_min, double a_0, double a_min, double v_0, double t_s, double t_v0, double t_n_a_min):

    return pos_reduced_jerk_a1(j_min, a_0, a_min, v_0, t_s, t_v0, t_n_a_min)


def pos_reduced_jerk_tv0_0_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min):

    return pos_reduced_jerk_tv0_0(j_min, a_0, a_min, v_0, p_0, p_max, t_s, t_n_a_min)


def pos_reduced_jerk_tv0_1_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min):

    return pos_reduced_jerk_tv0_1(j_min, a_0, a_min, v_0, p_0, p_max, t_s, t_n_a_min)


def pos_reduced_jerk_tv0_2_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min):

    return pos_reduced_jerk_tv0_2(j_min, a_0, a_min, v_0, p_0, p_max, t_s, t_n_a_min)


def pos_reduced_jerk_tv0_3_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min):

    return pos_reduced_jerk_tv0_3(j_min, a_0, a_min, v_0, p_0, p_max, t_s, t_n_a_min)


def pos_reduced_jerk_tv0_4_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min):

    return pos_reduced_jerk_tv0_4(j_min, a_0, a_min, v_0, p_0, p_max, t_s, t_n_a_min)


def pos_reduced_jerk_tv0_5_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min):

    return pos_reduced_jerk_tv0_5(j_min, a_0, a_min, v_0, p_0, p_max, t_s, t_n_a_min)


def pos_upper_bound_a1_max_0_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s):

    return pos_upper_bound_a1_max_0(j_min, a_0, a_min, v_0, p_0, p_max, t_s)


def pos_upper_bound_a1_max_1_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s):

    return pos_upper_bound_a1_max_1(j_min, a_0, a_min, v_0, p_0, p_max, t_s)


def pos_upper_bound_a1_max_2_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s):

    return pos_upper_bound_a1_max_2(j_min, a_0, a_min, v_0, p_0, p_max, t_s)


def pos_upper_bound_a1_max_3_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s):

    return pos_upper_bound_a1_max_3(j_min, a_0, a_min, v_0, p_0, p_max, t_s)


def pos_upper_bound_a1_max_4_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s):

    return pos_upper_bound_a1_max_4(j_min, a_0, a_min, v_0, p_0, p_max, t_s)


def pos_upper_bound_a1_max_5_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s):

    return pos_upper_bound_a1_max_5(j_min, a_0, a_min, v_0, p_0, p_max, t_s)


def pos_upper_bound_a1_min_0_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s):

    return pos_upper_bound_a1_min_0(j_min, a_0, a_min, v_0, p_0, p_max, t_s)


def pos_upper_bound_a1_min_1_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s):

    return pos_upper_bound_a1_min_1(j_min, a_0, a_min, v_0, p_0, p_max, t_s)


def pos_upper_bound_a1_min_2_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s):

    return pos_upper_bound_a1_min_2(j_min, a_0, a_min, v_0, p_0, p_max, t_s)


def pos_upper_bound_a1_min_3_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s):

    return pos_upper_bound_a1_min_3(j_min, a_0, a_min, v_0, p_0, p_max, t_s)


def pos_upper_bound_a1_min_4_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s):

    return pos_upper_bound_a1_min_4(j_min, a_0, a_min, v_0, p_0, p_max, t_s)


def pos_upper_bound_a1_min_5_c(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s):

    return pos_upper_bound_a1_min_5(j_min, a_0, a_min, v_0, p_0, p_max, t_s)


def pos_upper_bound_tv0_c(double j_min, double a_0, double a_1, double a_min, double v_0, double t_s):

    return pos_upper_bound_tv0(j_min, a_0, a_1, a_min, v_0, t_s)


def vel_reduced_a1_max_c(double j_min, double a_0, double a_n_plus_1_star, double v_0, double v_max, double t_s, double t_n):

    return vel_reduced_a1_max(j_min, a_0, a_n_plus_1_star, v_0, v_max, t_s, t_n)


def vel_reduced_a1_min_c(double j_min, double a_0, double a_n_plus_1_star, double v_0, double v_max, double t_s, double t_n):

    return vel_reduced_a1_min(j_min, a_0, a_n_plus_1_star, v_0, v_max, t_s, t_n)