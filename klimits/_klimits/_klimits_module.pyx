#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from cython.parallel cimport parallel
cimport openmp

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


cpdef normalize_np(np.ndarray[np.float64_t, ndim=1] value, np.ndarray[np.float64_t, ndim=2] value_range):
    normalized_value = -1 + 2 * (value - value_range[0]) / (value_range[1] - value_range[0])
    return normalized_value

cpdef normalize(double[::1] value, double[:, :] value_range):
    cdef int num_joints = value.shape[0]
    cdef int i
    cdef double[::1] normalized_value = np.zeros([num_joints], dtype=np.float64)
    for i in range(num_joints):
        normalized_value[i] = -1 + 2 * (value[i] - value_range[0, i]) / (value_range[1, i] - value_range[0, i])
    return np.asarray(normalized_value)

cpdef normalize_parallel(double[::1] value, double[:, :] value_range):
    cdef int num_joints = value.shape[0]
    cdef int i
    cdef double[::1] normalized_value = np.zeros([num_joints], dtype=np.float64)
    for i in prange(num_joints, nogil=True, schedule='static'):
        normalized_value[i] = -1 + 2 * (value[i] - value_range[0, i]) / (value_range[1, i] - value_range[0, i])
    return np.asarray(normalized_value)

cpdef normalize_batch_np(np.ndarray[np.float64_t, ndim=2] value, np.ndarray[np.float64_t, ndim=2] value_range):
    normalized_value = -1 + 2 * (value - value_range[0]) / (value_range[1] - value_range[0])
    return normalized_value

cpdef normalize_batch(double[:, :] value, double[:, :] value_range):
    cdef int num_steps = value.shape[0]
    cdef int num_joints = value.shape[1]
    cdef int i, j
    cdef double[:, ::1] normalized_value = np.zeros([num_steps, num_joints], dtype=np.float64)
    
    for i in range(num_steps):
        for j in range(num_joints):
            normalized_value[i, j] =  -1 + 2 * (value[i, j] - value_range[0, j]) / (value_range[1, j] - value_range[0, j])

    return np.asarray(normalized_value)

cpdef normalize_batch_parallel(double[:, :] value, double[:, :] value_range):
    cdef int num_steps = value.shape[0]
    cdef int num_joints = value.shape[1]
    cdef int i, j
    cdef double[:, ::1] normalized_value = np.zeros([num_steps, num_joints], dtype=np.float64)
    
    for i in prange(num_steps, nogil=True, schedule='static'):
        for j in range(num_joints):
            normalized_value[i, j] =  -1 + 2 * (value[i, j] - value_range[0, j]) / (value_range[1, j] - value_range[0, j])

    return np.asarray(normalized_value)

cpdef denormalize_np(np.ndarray[np.float64_t, ndim=1] norm_value,  np.ndarray[np.float64_t, ndim=2] value_range):
    actual_value = value_range[0] + 0.5 * (norm_value + 1) * (value_range[1] - value_range[0])
    return actual_value

cpdef denormalize(double[::1] norm_value, double[:, :] value_range):
    cdef int num_joints = norm_value.shape[0]
    cdef int i
    cdef double[::1] actual_value = np.zeros([num_joints], dtype=np.float64)
    for i in range(num_joints):
        actual_value[i] =  value_range[0, i] + 0.5 * (norm_value[i] + 1) * (value_range[1, i] - value_range[0, i])
    return np.asarray(actual_value)

cpdef denormalize_parallel(double[::1] norm_value, double[:, :] value_range):
    cdef int num_joints = norm_value.shape[0]
    cdef int i
    cdef double[::1] actual_value = np.zeros([num_joints], dtype=np.float64)
    for i in prange(num_joints, nogil=True, schedule='static'):
        actual_value[i] =  value_range[0, i] + 0.5 * (norm_value[i] + 1) * (value_range[1, i] - value_range[0, i])
    return np.asarray(actual_value)

cpdef denormalize_batch_np(np.ndarray[np.float64_t, ndim=2] norm_value, np.ndarray[np.float64_t, ndim=2] value_range):
    actual_value = value_range[0] + 0.5 * (norm_value + 1) * (value_range[1] - value_range[0])
    return actual_value

cpdef denormalize_batch(double[:, :] norm_value, double[:, :] value_range):
    cdef int num_steps = norm_value.shape[0]
    cdef int num_joints = norm_value.shape[1]
    cdef int i, j
    cdef double[:, ::1] actual_value = np.zeros([num_steps, num_joints], dtype=np.float64)
    
    for i in range(num_steps):
        for j in range(num_joints):
            actual_value[i, j] = value_range[0, j] + 0.5 * (norm_value[i, j] + 1) * (value_range[1, j] - value_range[0, j])
            
    return np.asarray(actual_value)

cpdef denormalize_batch_parallel(double[:, :] norm_value, double[:, :] value_range):
    cdef int num_steps = norm_value.shape[0]
    cdef int num_joints = norm_value.shape[1]
    cdef int i, j
    cdef double[:, ::1] actual_value = np.zeros([num_steps, num_joints], dtype=np.float64)
    
    for i in prange(num_steps, nogil=True, schedule='static'):
        for j in range(num_joints):
            actual_value[i, j] = value_range[0, j] + 0.5 * (norm_value[i, j] + 1) * (value_range[1, j] - value_range[0, j])

    return np.asarray(actual_value)

cpdef calculate_end_position_np(np.ndarray[np.float64_t, ndim=1] start_acceleration, np.ndarray[np.float64_t, ndim=1] end_acceleration, 
                                np.ndarray[np.float64_t, ndim=1] start_velocity, np.ndarray[np.float64_t, ndim=1] start_position,
                                double trajectory_time_step):
    end_position = start_position + start_velocity * trajectory_time_step + \
                   (0.33333333333333333 * start_acceleration + 0.16666666666666666 * end_acceleration) * trajectory_time_step ** 2                   
  
    return end_position

cpdef calculate_end_position(double[::1] start_acceleration, double[::1] end_acceleration, double[::1] start_velocity, 
                             double[::1] start_position, double trajectory_time_step):
    cdef int num_joints = start_acceleration.shape[0]     
    cdef int i
    cdef double[::1] end_position = np.zeros([num_joints], dtype=np.float64)
    for i in range(num_joints):     
        end_position[i] = start_position[i] + start_velocity[i] * trajectory_time_step + \
                          (0.33333333333333333 * start_acceleration[i] + 0.16666666666666666 * end_acceleration[i]) * trajectory_time_step ** 2                   
  
    return np.asarray(end_position)

cpdef calculate_end_position_parallel(double[::1] start_acceleration, double[::1] end_acceleration, double[::1] start_velocity, 
                                      double[::1] start_position, double trajectory_time_step):
    cdef int num_joints = start_acceleration.shape[0]     
    cdef int i
    cdef double[::1] end_position = np.zeros([num_joints], dtype=np.float64)
    for i in prange(num_joints, nogil=True, schedule='static'):     
        end_position[i] = start_position[i] + start_velocity[i] * trajectory_time_step + \
                          (0.33333333333333333 * start_acceleration[i] + 0.16666666666666666 * end_acceleration[i]) * trajectory_time_step ** 2                   
  
    return np.asarray(end_position)

cpdef interpolate_position_np(np.ndarray[np.float64_t, ndim=1] start_acceleration, np.ndarray[np.float64_t, ndim=1] end_acceleration, 
                              np.ndarray[np.float64_t, ndim=1] start_velocity, np.ndarray[np.float64_t, ndim=1] start_position,
                              double time_since_start, double trajectory_time_step):
    cdef int num_joints = start_acceleration.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] interpolated_position
    interpolated_position = start_position + start_velocity * time_since_start + \
                            0.5 * start_acceleration * time_since_start ** 2 + \
                            0.16666666666666666 * ((end_acceleration - start_acceleration) / trajectory_time_step) * time_since_start ** 3                   
  
    return interpolated_position

cpdef interpolate_position(double[::1] start_acceleration, double[::1] end_acceleration, 
                           double[::1] start_velocity, double[::1] start_position,
                           double time_since_start, double trajectory_time_step):
    cdef int num_joints = start_acceleration.shape[0]     
    cdef int i
    cdef double[::1] interpolated_position = np.zeros([num_joints], dtype=np.float64)
    for i in range(num_joints):     
        interpolated_position[i] = start_position[i] + start_velocity[i] * time_since_start + \
                                   0.5 * start_acceleration[i] * time_since_start ** 2 + \
                                   0.16666666666666666 * ((end_acceleration[i] - start_acceleration[i]) / trajectory_time_step) * time_since_start ** 3

    return np.asarray(interpolated_position)

cpdef interpolate_position_parallel(double[::1] start_acceleration, double[::1] end_acceleration, 
                                    double[::1] start_velocity, double[::1] start_position,
                                    double time_since_start, double trajectory_time_step):
    cdef int num_joints = start_acceleration.shape[0]     
    cdef int i
    cdef double[::1] interpolated_position = np.zeros([num_joints], dtype=np.float64)
    for i in prange(num_joints, nogil=True, schedule='static'):     
        interpolated_position[i] = start_position[i] + start_velocity[i] * time_since_start + \
                                   0.5 * start_acceleration[i] * time_since_start ** 2 + \
                                   0.16666666666666666 * ((end_acceleration[i] - start_acceleration[i]) / trajectory_time_step) * time_since_start ** 3

    return np.asarray(interpolated_position)

cpdef interpolate_position_batch_np(np.ndarray[np.float64_t, ndim=1] start_acceleration, np.ndarray[np.float64_t, ndim=1] end_acceleration, 
                                    np.ndarray[np.float64_t, ndim=1] start_velocity, np.ndarray[np.float64_t, ndim=1] start_position,
                                    np.ndarray[np.float64_t, ndim=1] time_since_start, double trajectory_time_step):
    cdef int num_steps = time_since_start.shape[0]
    cdef int num_joints = start_acceleration.shape[0]
    cdef int i
    cdef np.ndarray[np.float64_t, ndim=2] interpolated_position = np.zeros([num_steps, num_joints], dtype=np.float64)
    
    for i in range(num_steps):
        interpolated_position[i] = start_position + start_velocity * time_since_start[i] + \
                                0.5 * start_acceleration * time_since_start[i] ** 2 + \
                                0.16666666666666666 * ((end_acceleration - start_acceleration)
                                            / trajectory_time_step) * time_since_start[i] ** 3

    return interpolated_position

cpdef interpolate_position_batch(double[::1] start_acceleration, double[::1] end_acceleration, 
                                 double[::1] start_velocity, double[::1] start_position,
                                 double[::1] time_since_start, double trajectory_time_step):
    cdef int num_steps = time_since_start.shape[0]
    cdef int num_joints = start_acceleration.shape[0]
    cdef int i, j
    cdef double[:, ::1] interpolated_position = np.zeros([num_steps, num_joints], dtype=np.float64)
    
    for i in range(num_steps):
        for j in range(num_joints):
            interpolated_position[i, j] = start_position[j] + start_velocity[j] * time_since_start[i] + \
                                          0.5 * start_acceleration[j] * time_since_start[i] ** 2 + \
                                          0.16666666666666666 * ((end_acceleration[j] - start_acceleration[j]) / 
                                          trajectory_time_step) * time_since_start[i] ** 3

    return np.asarray(interpolated_position)

cpdef interpolate_position_batch_parallel(double[::1] start_acceleration, double[::1] end_acceleration, 
                                          double[::1] start_velocity, double[::1] start_position,
                                          double[::1] time_since_start, double trajectory_time_step):
    cdef int num_steps = time_since_start.shape[0]
    cdef int num_joints = start_acceleration.shape[0]
    cdef int i, j
    cdef double[:, ::1] interpolated_position = np.zeros([num_steps, num_joints], dtype=np.float64)
    
    for i in prange(num_steps, nogil=True, schedule='static'):
        for j in range(num_joints):
            interpolated_position[i, j] = start_position[j] + start_velocity[j] * time_since_start[i] + \
                                          0.5 * start_acceleration[j] * time_since_start[i] ** 2 + \
                                          0.16666666666666666 * ((end_acceleration[j] - start_acceleration[j]) / 
                                          trajectory_time_step) * time_since_start[i] ** 3

    return np.asarray(interpolated_position)

cpdef calculate_end_velocity_np(np.ndarray[np.float64_t, ndim=1] start_acceleration, np.ndarray[np.float64_t, ndim=1] end_acceleration, 
                                np.ndarray[np.float64_t, ndim=1] start_velocity, double trajectory_time_step):

    end_velocity = start_velocity + 0.5 * (start_acceleration + end_acceleration) * trajectory_time_step 

    return end_velocity

cpdef calculate_end_velocity(double[::1] start_acceleration, double[::1] end_acceleration, 
                             double[::1] start_velocity, double trajectory_time_step):

    cdef int num_joints = start_acceleration.shape[0]     
    cdef int i
    cdef double[::1] end_velocity = np.zeros([num_joints], dtype=np.float64)
    for i in range(num_joints): 
        end_velocity[i] = start_velocity[i] + 0.5 * (start_acceleration[i] + end_acceleration[i]) * trajectory_time_step 

    return np.asarray(end_velocity)

cpdef calculate_end_velocity_parallel(double[::1] start_acceleration, double[::1] end_acceleration, 
                                      double[::1] start_velocity, double trajectory_time_step):

    cdef int num_joints = start_acceleration.shape[0]     
    cdef int i
    cdef double[::1] end_velocity = np.zeros([num_joints], dtype=np.float64)
    for i in prange(num_joints, nogil=True, schedule='static'): 
        end_velocity[i] = start_velocity[i] + 0.5 * (start_acceleration[i] + end_acceleration[i]) * trajectory_time_step 

    return np.asarray(end_velocity)


cpdef interpolate_velocity_np(np.ndarray[np.float64_t, ndim=1] start_acceleration, np.ndarray[np.float64_t, ndim=1] end_acceleration, 
                              np.ndarray[np.float64_t, ndim=1] start_velocity, double time_since_start, double trajectory_time_step):

    interpolated_velocity = start_velocity + start_acceleration * time_since_start + \
                            0.5 * ((end_acceleration - start_acceleration) /
                                    trajectory_time_step) * time_since_start ** 2

    return interpolated_velocity

cpdef interpolate_velocity(double[::1] start_acceleration, double[::1] end_acceleration, 
                           double[::1] start_velocity, double time_since_start, double trajectory_time_step):
    cdef int num_joints = start_acceleration.shape[0]     
    cdef int i
    cdef double[::1] interpolated_velocity = np.zeros([num_joints], dtype=np.float64)
    for i in range(num_joints):                           
        interpolated_velocity[i] = start_velocity[i] + start_acceleration[i] * time_since_start + \
                                   0.5 * ((end_acceleration[i] - start_acceleration[i]) /
                                        trajectory_time_step) * time_since_start ** 2

    return np.asarray(interpolated_velocity)

cpdef interpolate_velocity_parallel(double[::1] start_acceleration, double[::1] end_acceleration, 
                                    double[::1] start_velocity, double time_since_start, double trajectory_time_step):
    cdef int num_joints = start_acceleration.shape[0]     
    cdef int i
    cdef double[::1] interpolated_velocity = np.zeros([num_joints], dtype=np.float64)
    for i in prange(num_joints, nogil=True, schedule='static'):                           
        interpolated_velocity[i] = start_velocity[i] + start_acceleration[i] * time_since_start + \
                                   0.5 * ((end_acceleration[i] - start_acceleration[i]) /
                                        trajectory_time_step) * time_since_start ** 2

    return np.asarray(interpolated_velocity)

cpdef interpolate_velocity_batch_np(np.ndarray[np.float64_t, ndim=1] start_acceleration, np.ndarray[np.float64_t, ndim=1] end_acceleration, 
                                    np.ndarray[np.float64_t, ndim=1] start_velocity, np.ndarray[np.float64_t, ndim=1] time_since_start, double trajectory_time_step):

    cdef int num_steps = time_since_start.shape[0]
    cdef int num_joints = start_acceleration.shape[0]
    cdef int i
    cdef np.ndarray[np.float64_t, ndim=2] interpolated_velocity = np.zeros([num_steps, num_joints], dtype=np.float64)
    
    for i in range(num_steps):
        interpolated_velocity[i] = start_velocity + start_acceleration * time_since_start[i] + \
                                   0.5 * ((end_acceleration - start_acceleration) /
                                   trajectory_time_step) * time_since_start[i] ** 2

    return interpolated_velocity

cpdef interpolate_velocity_batch(double[::1] start_acceleration, double[::1] end_acceleration, 
                                 double[::1] start_velocity, double[::1] time_since_start, double trajectory_time_step):

    cdef int num_steps = time_since_start.shape[0]
    cdef int num_joints = start_acceleration.shape[0]
    cdef int i, j
    cdef double[:, ::1] interpolated_velocity = np.zeros([num_steps, num_joints], dtype=np.float64)
    
    for i in range(num_steps):
        for j in range(num_joints):
            interpolated_velocity[i, j] = start_velocity[j]+ start_acceleration[j] * time_since_start[i] + \
                                   0.5 * ((end_acceleration[j] - start_acceleration[j]) /
                                   trajectory_time_step) * time_since_start[i] ** 2

    return np.asarray(interpolated_velocity)

cpdef interpolate_velocity_batch_parallel(double[::1] start_acceleration, double[::1] end_acceleration, 
                                          double[::1] start_velocity, double[::1] time_since_start, double trajectory_time_step):

    cdef int num_steps = time_since_start.shape[0]
    cdef int num_joints = start_acceleration.shape[0]
    cdef int i, j
    cdef double[:, ::1] interpolated_velocity = np.zeros([num_steps, num_joints], dtype=np.float64)
    
    for i in prange(num_steps, nogil=True, schedule='static'):
        for j in range(num_joints):
            interpolated_velocity[i, j] = start_velocity[j]+ start_acceleration[j] * time_since_start[i] + \
                                   0.5 * ((end_acceleration[j] - start_acceleration[j]) /
                                   trajectory_time_step) * time_since_start[i] ** 2

    return np.asarray(interpolated_velocity)

cpdef interpolate_acceleration_np(np.ndarray[np.float64_t, ndim=1] start_acceleration, np.ndarray[np.float64_t, ndim=1] end_acceleration, 
                                  double time_since_start, double trajectory_time_step):
    interpolated_acceleration = start_acceleration + ((end_acceleration - start_acceleration) /
                                                        trajectory_time_step) * time_since_start

    return interpolated_acceleration

cpdef interpolate_acceleration(double[::1] start_acceleration, double[::1] end_acceleration, 
                               double time_since_start, double trajectory_time_step):
    cdef int num_joints = start_acceleration.shape[0]     
    cdef int i
    cdef double[::1] interpolated_acceleration = np.zeros([num_joints], dtype=np.float64)
    for i in range(num_joints):    
        interpolated_acceleration[i] = start_acceleration[i] + ((end_acceleration[i] - start_acceleration[i]) /
                                    trajectory_time_step) * time_since_start

    return np.asarray(interpolated_acceleration)

cpdef interpolate_acceleration_parallel(double[::1] start_acceleration, double[::1] end_acceleration, 
                                        double time_since_start, double trajectory_time_step):
    cdef int num_joints = start_acceleration.shape[0]     
    cdef int i
    cdef double[::1] interpolated_acceleration = np.zeros([num_joints], dtype=np.float64)
    for i in prange(num_joints, nogil=True, schedule='static'):    
        interpolated_acceleration[i] = start_acceleration[i] + ((end_acceleration[i] - start_acceleration[i]) /
                                    trajectory_time_step) * time_since_start

    return np.asarray(interpolated_acceleration)

cpdef interpolate_acceleration_batch_np(np.ndarray[np.float64_t, ndim=1] start_acceleration, np.ndarray[np.float64_t, ndim=1] end_acceleration, 
                                        np.ndarray[np.float64_t, ndim=1] time_since_start, double trajectory_time_step):
    cdef int num_steps = time_since_start.shape[0]
    cdef int num_joints = start_acceleration.shape[0]
    cdef int i
    cdef np.ndarray[np.float64_t, ndim=2] interpolated_acceleration = np.zeros([num_steps, num_joints], dtype=np.float64)
    
    for i in range(num_steps):
        interpolated_acceleration[i] = start_acceleration + ((end_acceleration - start_acceleration) /
                                                            trajectory_time_step) * time_since_start[i]

    return interpolated_acceleration

cpdef interpolate_acceleration_batch(double[::1] start_acceleration, double[::1] end_acceleration, 
                                     double[::1] time_since_start, double trajectory_time_step):
    cdef int num_steps = time_since_start.shape[0]
    cdef int num_joints = start_acceleration.shape[0]
    cdef int i, j
    cdef double[:, ::1] interpolated_acceleration = np.zeros([num_steps, num_joints], dtype=np.float64)
    
    for i in range(num_steps):
        for j in range(num_joints):
            interpolated_acceleration[i, j] = start_acceleration[j] + ((end_acceleration[j] - start_acceleration[j]) /
                                              trajectory_time_step) * time_since_start[i]

    return np.asarray(interpolated_acceleration)

cpdef interpolate_acceleration_batch_parallel(double[::1] start_acceleration, double[::1] end_acceleration, 
                                              double[::1] time_since_start, double trajectory_time_step):
    cdef int num_steps = time_since_start.shape[0]
    cdef int num_joints = start_acceleration.shape[0]
    cdef int i, j
    cdef double[:, ::1] interpolated_acceleration = np.zeros([num_steps, num_joints], dtype=np.float64)
    
    for i in prange(num_steps, nogil=True, schedule='static'):
        for j in range(num_joints):
            interpolated_acceleration[i, j] = start_acceleration[j] + ((end_acceleration[j] - start_acceleration[j]) /
                                              trajectory_time_step) * time_since_start[i]

    return np.asarray(interpolated_acceleration)

cpdef get_num_threads():
    cdef int num_threads, i
    for i in prange(1, nogil=True):
        num_threads = openmp.omp_get_num_threads()
    return num_threads