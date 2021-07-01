#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from cython.parallel cimport parallel
from cython.view cimport array as cvarray
cimport openmp
from libc.math cimport fabs, floor, ceil, round, isnan, fmax, fmin, sqrt

cdef extern from '_klimits_code.h':
    double pos_all_a1_max(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min) nogil
    double pos_all_a1_min(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min) nogil
    double pos_all_bounded_vel_continuous_a1(double j_min, double j_max, double a_0, double a_min, double v_0, double t_s, double t_star, double t_n_a_min, double t_u) nogil
    double complex pos_all_bounded_vel_continuous_tu(double j_min, double j_max, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_star, double t_n_a_min) nogil
    double pos_all_bounded_vel_discrete_a1(double j_min, double j_max, double j_n_u_plus_1, double a_0, double a_min, double v_0, double t_s, double t_star, double t_n_a_min, double t_n_u) nogil
    double pos_all_bounded_vel_discrete_j_n_u_plus_1(double j_min, double j_max, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_star, double t_n_a_min, double t_n_u) nogil
    double pos_all_tv0(double j_min, double a_0, double a_1, double a_min, double v_0, double t_s, double t_n_a_min) nogil
    double pos_first_a1(double a_0, double v_0, double t_s, double t_v0) nogil
    double pos_first_tv0_max(double a_0, double v_0, double p_0, double p_max) nogil
    double pos_first_tv0_min(double a_0, double v_0, double p_0, double p_max) nogil
    double pos_min_jerk_a1(double j_min, double a_0, double v_0, double t_s, double t_v0) nogil
    double pos_min_jerk_bounded_vel_continuous_a1(double j_min, double j_max, double a_0, double v_0, double t_s, double t_star, double t_u) nogil
    double complex pos_min_jerk_bounded_vel_continuous_tu(double j_min, double j_max, double a_0, double v_0, double p_0, double p_max, double t_s, double t_star) nogil
    double pos_min_jerk_bounded_vel_discrete_a1(double j_min, double j_max, double j_n_u_plus_1, double a_0, double v_0, double t_s, double t_star, double t_n_u) nogil
    double pos_min_jerk_bounded_vel_discrete_j_n_u_plus_1(double j_min, double j_max, double a_0, double v_0, double p_0, double p_max, double t_s, double t_star, double t_n_u) nogil
    double pos_min_jerk_tv0_0(double j_min, double a_0, double v_0, double p_0, double p_max, double t_s) nogil
    double pos_min_jerk_tv0_1(double j_min, double a_0, double v_0, double p_0, double p_max, double t_s) nogil
    double complex pos_min_jerk_tv0_2(double j_min, double a_0, double v_0, double p_0, double p_max, double t_s) nogil
    double pos_reduced_jerk_a1(double j_min, double a_0, double a_min, double v_0, double t_s, double t_v0, double t_n_a_min) nogil
    double pos_reduced_jerk_tv0_0(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min) nogil
    double pos_reduced_jerk_tv0_1(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min) nogil
    double complex pos_reduced_jerk_tv0_2(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min) nogil
    double pos_reduced_jerk_tv0_3(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min) nogil
    double pos_reduced_jerk_tv0_4(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min) nogil
    double complex pos_reduced_jerk_tv0_5(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s, double t_n_a_min) nogil 
    double pos_upper_bound_a1_max_0(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s) nogil
    double pos_upper_bound_a1_max_1(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s) nogil
    double complex pos_upper_bound_a1_max_2(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s) nogil
    double pos_upper_bound_a1_max_3(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s) nogil
    double pos_upper_bound_a1_max_4(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s) nogil
    double complex pos_upper_bound_a1_max_5(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s) nogil
    double pos_upper_bound_a1_min_0(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s) nogil
    double pos_upper_bound_a1_min_1(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s) nogil
    double complex pos_upper_bound_a1_min_2(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s) nogil
    double pos_upper_bound_a1_min_3(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s) nogil
    double pos_upper_bound_a1_min_4(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s) nogil
    double complex pos_upper_bound_a1_min_5(double j_min, double a_0, double a_min, double v_0, double p_0, double p_max, double t_s) nogil
    double pos_upper_bound_tv0(double j_min, double a_0, double a_1, double a_min, double v_0, double t_s) nogil
    double vel_zero_a_n_plus_1_star_a1(double j_min, double a_0, double v_0, double v_max, double t_s, double t_n) nogil
    double vel_fixed_a_n_plus_1_star_a1_max(double j_min, double a_0, double a_n_plus_1_star, double v_0, double v_max, double t_s, double t_n) nogil
    double vel_fixed_a_n_plus_1_star_a1_min(double j_min, double a_0, double a_n_plus_1_star, double v_0, double v_max, double t_s, double t_n) nogil
    double compute_distance(double pos_a_0, double pos_a_1, double pos_a_2, double pos_b_0, double pos_b_1, double pos_b_2, double radius_a, double radius_b) nogil


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

def vel_zero_a_n_plus_1_star_a1_c(double j_min, double a_0, double v_0, double v_max, double t_s, double t_n):

    return vel_zero_a_n_plus_1_star_a1(j_min, a_0, v_0, v_max, t_s, t_n)


def vel_fixed_a_n_plus_1_star_a1_max_c(double j_min, double a_0, double a_n_plus_1_star, double v_0, double v_max, double t_s, double t_n):

    return vel_fixed_a_n_plus_1_star_a1_max(j_min, a_0, a_n_plus_1_star, v_0, v_max, t_s, t_n)


def vel_fixed_a_n_plus_1_star_a1_min_c(double j_min, double a_0, double a_n_plus_1_star, double v_0, double v_max, double t_s, double t_n):

    return vel_fixed_a_n_plus_1_star_a1_min(j_min, a_0, a_n_plus_1_star, v_0, v_max, t_s, t_n)

def compute_distance_c(double pos_a_0, double pos_a_1, double pos_a_2, double pos_b_0, double pos_b_1, double pos_b_2, double radius_a, double radius_b):
    return compute_distance(pos_a_0, pos_a_1, pos_a_2, pos_b_0, pos_b_1, pos_b_2, radius_a, radius_b)


cpdef normalize_np(np.ndarray[np.float64_t, ndim=1] value, np.ndarray[np.float64_t, ndim=2] value_range):
    normalized_value = -1 + 2 * (value - value_range[0]) / (value_range[1] - value_range[0])
    return normalized_value

cpdef normalize(double[::1] value, double[:, :] value_range):
    cdef int num_joints = value.shape[0]
    cdef int i
    cdef double[::1] normalized_value = np.empty([num_joints], dtype=np.float64)
    for i in range(num_joints):
        normalized_value[i] = -1 + 2 * (value[i] - value_range[0, i]) / (value_range[1, i] - value_range[0, i])
    return np.asarray(normalized_value)

cpdef normalize_parallel(double[::1] value, double[:, :] value_range):
    cdef int num_joints = value.shape[0]
    cdef int i
    cdef double[::1] normalized_value = np.empty([num_joints], dtype=np.float64)
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
    cdef double[:, ::1] normalized_value = np.empty([num_steps, num_joints], dtype=np.float64)
    
    for i in range(num_steps):
        for j in range(num_joints):
            normalized_value[i, j] =  -1 + 2 * (value[i, j] - value_range[0, j]) / (value_range[1, j] - value_range[0, j])

    return np.asarray(normalized_value)

cpdef normalize_batch_parallel(double[:, :] value, double[:, :] value_range):
    cdef int num_steps = value.shape[0]
    cdef int num_joints = value.shape[1]
    cdef int i, j
    cdef double[:, ::1] normalized_value = np.empty([num_steps, num_joints], dtype=np.float64)
    
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
    cdef double[::1] actual_value = np.empty([num_joints], dtype=np.float64)
    for i in range(num_joints):
        actual_value[i] =  value_range[0, i] + 0.5 * (norm_value[i] + 1) * (value_range[1, i] - value_range[0, i])
    return np.asarray(actual_value)

cpdef denormalize_parallel(double[::1] norm_value, double[:, :] value_range):
    cdef int num_joints = norm_value.shape[0]
    cdef int i
    cdef double[::1] actual_value = np.empty([num_joints], dtype=np.float64)
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
    cdef double[:, ::1] actual_value = np.empty([num_steps, num_joints], dtype=np.float64)
    
    for i in range(num_steps):
        for j in range(num_joints):
            actual_value[i, j] = value_range[0, j] + 0.5 * (norm_value[i, j] + 1) * (value_range[1, j] - value_range[0, j])
            
    return np.asarray(actual_value)

cpdef denormalize_batch_parallel(double[:, :] norm_value, double[:, :] value_range):
    cdef int num_steps = norm_value.shape[0]
    cdef int num_joints = norm_value.shape[1]
    cdef int i, j
    cdef double[:, ::1] actual_value = np.empty([num_steps, num_joints], dtype=np.float64)
    
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
    cdef double[::1] end_position = np.empty([num_joints], dtype=np.float64)
    for i in range(num_joints):     
        end_position[i] = start_position[i] + start_velocity[i] * trajectory_time_step + \
                          (0.33333333333333333 * start_acceleration[i] + 0.16666666666666666 * end_acceleration[i]) * trajectory_time_step ** 2                   
  
    return np.asarray(end_position)

cpdef calculate_end_position_parallel(double[::1] start_acceleration, double[::1] end_acceleration, double[::1] start_velocity, 
                                      double[::1] start_position, double trajectory_time_step):
    cdef int num_joints = start_acceleration.shape[0]     
    cdef int i
    cdef double[::1] end_position = np.empty([num_joints], dtype=np.float64)
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
    cdef double[::1] interpolated_position = np.empty([num_joints], dtype=np.float64)
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
    cdef double[::1] interpolated_position = np.empty([num_joints], dtype=np.float64)
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
    cdef np.ndarray[np.float64_t, ndim=2] interpolated_position = np.empty([num_steps, num_joints], dtype=np.float64)
    
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
    cdef double[:, ::1] interpolated_position = np.empty([num_steps, num_joints], dtype=np.float64)
    
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
    cdef double[:, ::1] interpolated_position = np.empty([num_steps, num_joints], dtype=np.float64)
    
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
    cdef double[::1] end_velocity = np.empty([num_joints], dtype=np.float64)
    for i in range(num_joints): 
        end_velocity[i] = start_velocity[i] + 0.5 * (start_acceleration[i] + end_acceleration[i]) * trajectory_time_step 

    return np.asarray(end_velocity)

cpdef calculate_end_velocity_parallel(double[::1] start_acceleration, double[::1] end_acceleration, 
                                      double[::1] start_velocity, double trajectory_time_step):

    cdef int num_joints = start_acceleration.shape[0]     
    cdef int i
    cdef double[::1] end_velocity = np.empty([num_joints], dtype=np.float64)
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
    cdef double[::1] interpolated_velocity = np.empty([num_joints], dtype=np.float64)
    for i in range(num_joints):                           
        interpolated_velocity[i] = start_velocity[i] + start_acceleration[i] * time_since_start + \
                                   0.5 * ((end_acceleration[i] - start_acceleration[i]) /
                                        trajectory_time_step) * time_since_start ** 2

    return np.asarray(interpolated_velocity)

cpdef interpolate_velocity_parallel(double[::1] start_acceleration, double[::1] end_acceleration, 
                                    double[::1] start_velocity, double time_since_start, double trajectory_time_step):
    cdef int num_joints = start_acceleration.shape[0]     
    cdef int i
    cdef double[::1] interpolated_velocity = np.empty([num_joints], dtype=np.float64)
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
    cdef np.ndarray[np.float64_t, ndim=2] interpolated_velocity = np.empty([num_steps, num_joints], dtype=np.float64)
    
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
    cdef double[:, ::1] interpolated_velocity = np.empty([num_steps, num_joints], dtype=np.float64)
    
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
    cdef double[:, ::1] interpolated_velocity = np.empty([num_steps, num_joints], dtype=np.float64)
    
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
    cdef double[::1] interpolated_acceleration = np.empty([num_joints], dtype=np.float64)
    for i in range(num_joints):    
        interpolated_acceleration[i] = start_acceleration[i] + ((end_acceleration[i] - start_acceleration[i]) /
                                    trajectory_time_step) * time_since_start

    return np.asarray(interpolated_acceleration)

cpdef interpolate_acceleration_parallel(double[::1] start_acceleration, double[::1] end_acceleration, 
                                        double time_since_start, double trajectory_time_step):
    cdef int num_joints = start_acceleration.shape[0]     
    cdef int i
    cdef double[::1] interpolated_acceleration = np.empty([num_joints], dtype=np.float64)
    for i in prange(num_joints, nogil=True, schedule='static'):    
        interpolated_acceleration[i] = start_acceleration[i] + ((end_acceleration[i] - start_acceleration[i]) /
                                    trajectory_time_step) * time_since_start

    return np.asarray(interpolated_acceleration)

cpdef interpolate_acceleration_batch_np(np.ndarray[np.float64_t, ndim=1] start_acceleration, np.ndarray[np.float64_t, ndim=1] end_acceleration, 
                                        np.ndarray[np.float64_t, ndim=1] time_since_start, double trajectory_time_step):
    cdef int num_steps = time_since_start.shape[0]
    cdef int num_joints = start_acceleration.shape[0]
    cdef int i
    cdef np.ndarray[np.float64_t, ndim=2] interpolated_acceleration = np.empty([num_steps, num_joints], dtype=np.float64)
    
    for i in range(num_steps):
        interpolated_acceleration[i] = start_acceleration + ((end_acceleration - start_acceleration) /
                                                            trajectory_time_step) * time_since_start[i]

    return interpolated_acceleration

cpdef interpolate_acceleration_batch(double[::1] start_acceleration, double[::1] end_acceleration, 
                                     double[::1] time_since_start, double trajectory_time_step):
    cdef int num_steps = time_since_start.shape[0]
    cdef int num_joints = start_acceleration.shape[0]
    cdef int i, j
    cdef double[:, ::1] interpolated_acceleration = np.empty([num_steps, num_joints], dtype=np.float64)
    
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
    cdef double[:, ::1] interpolated_acceleration = np.empty([num_steps, num_joints], dtype=np.float64)
    
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


cdef double nan = float('nan')


cdef class PosVelJerkLimitation:
    cdef double _time_step 
    cdef int _num_joints, 
    cdef object _num_threads
    cdef double[:, :] _jerk_limits, _acc_limits_min_max, _acc_limits_after_max_vel, _acc_limits, _vel_limits, _pos_limits
    cdef bint _set_velocity_after_max_pos_to_zero, _limit_velocity, _limit_position, _soft_velocity_limits, _soft_position_limits, _normalize_acc_range

    def __init__(self,
                 double time_step,
                 object pos_limits,
                 object vel_limits,
                 object acc_limits,
                 object jerk_limits,
                 double acceleration_after_max_vel_limit_factor=0.0001,
                 bint set_velocity_after_max_pos_to_zero=True,
                 bint limit_velocity=True,
                 bint limit_position=True,
                 object num_threads=None,
                 bint soft_velocity_limits=False,
                 bint soft_position_limits=False,
                 bint normalize_acc_range=True,
                 *vargs,
                 **kwargs):

        self._time_step = time_step
        self._vel_limits = np.asarray(vel_limits, dtype=np.float64)
        self._num_joints = len(self._vel_limits)
        self._pos_limits = np.asarray(pos_limits, dtype=np.float64) if pos_limits is not None \
            else np.full(shape=(self._num_joints, 2), fill_value=nan, dtype=np.float64)
        self._acc_limits = np.asarray(acc_limits, dtype=np.float64)
        self._acc_limits_min_max = np.swapaxes(acc_limits, 0, 1)
        self._jerk_limits = np.asarray(jerk_limits, dtype=np.float64)

        self._acc_limits_after_max_vel = np.multiply(self._acc_limits, acceleration_after_max_vel_limit_factor)

        self._set_velocity_after_max_pos_to_zero = set_velocity_after_max_pos_to_zero
        self._limit_velocity = limit_velocity
        self._limit_position = limit_position
        self._soft_velocity_limits = soft_velocity_limits
        self._soft_position_limits = soft_position_limits
        self._normalize_acc_range = normalize_acc_range

        self._num_threads = num_threads

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

    def calculate_valid_acceleration_range(self, double[:] current_pos, double[:] current_vel, double[:] current_acc, bint braking_trajectory=False,
                                           int time_step_counter=0, double[:, :] limit_min_max=None, **kwargs):
        if limit_min_max is None:
            limit_min_max = np.ones(shape=(self._num_joints, 2))
        if current_pos is None:
            current_pos = np.empty(shape=self._num_joints)

        cdef double[:, :] acc_range = np.empty((self._num_joints, 2))
        cdef double[:, :] acc_range_min_max
        cdef double[:] limit_violation = np.empty(self._num_joints)
        cdef double[:, :] acc_range_dynamic_vel = self._acc_limits.copy()
        cdef double[:, :] acc_range_dynamic_pos = np.empty((self._num_joints, 2))

        cdef int i, limit_violation_joint, num_threads
        cdef double acc_range_min, acc_range_max

        if self._num_threads is None:
            for i in prange(self._num_joints, nogil=True):
                acc_range_min, acc_range_max, limit_violation_joint = \
                    calculate_valid_acceleration_range_per_joint(i, self._time_step, current_pos[i], current_vel[i],
                                                                 current_acc[i], self._pos_limits[i],
                                                                 self._vel_limits[i], self._acc_limits[i],
                                                                 self._jerk_limits[i],
                                                                 self._acc_limits_after_max_vel[i],
                                                                 self._set_velocity_after_max_pos_to_zero,
                                                                 self._limit_velocity, self._limit_position,
                                                                 braking_trajectory, time_step_counter,
                                                                 limit_min_max[i], self._soft_velocity_limits,
                                                                 self._soft_position_limits,
                                                                 acc_range_dynamic_vel[i],
                                                                 acc_range_dynamic_pos[i])

                acc_range[i][0] = acc_range_min
                acc_range[i][1] = acc_range_max
                limit_violation[i] = limit_violation_joint
        else:
            num_threads = int(self._num_threads)
            if num_threads <= 1:
                for i in range(self._num_joints):
                    acc_range_min, acc_range_max, limit_violation_joint = \
                        calculate_valid_acceleration_range_per_joint(i, self._time_step, current_pos[i], current_vel[i],
                                                                     current_acc[i], self._pos_limits[i],
                                                                     self._vel_limits[i], self._acc_limits[i],
                                                                     self._jerk_limits[i],
                                                                     self._acc_limits_after_max_vel[i],
                                                                     self._set_velocity_after_max_pos_to_zero,
                                                                     self._limit_velocity, self._limit_position,
                                                                     braking_trajectory, time_step_counter,
                                                                     limit_min_max[i], self._soft_velocity_limits,
                                                                     self._soft_position_limits,
                                                                     acc_range_dynamic_vel[i],
                                                                     acc_range_dynamic_pos[i])

                    acc_range[i][0] = acc_range_min
                    acc_range[i][1] = acc_range_max
                    limit_violation[i] = limit_violation_joint
            else:
                for i in prange(self._num_joints, nogil=True, num_threads=num_threads):
                    acc_range_min, acc_range_max, limit_violation_joint = \
                        calculate_valid_acceleration_range_per_joint(i, self._time_step, current_pos[i], current_vel[i],
                                                                     current_acc[i], self._pos_limits[i],
                                                                     self._vel_limits[i], self._acc_limits[i],
                                                                     self._jerk_limits[i],
                                                                     self._acc_limits_after_max_vel[i],
                                                                     self._set_velocity_after_max_pos_to_zero,
                                                                     self._limit_velocity, self._limit_position,
                                                                     braking_trajectory, time_step_counter,
                                                                     limit_min_max[i], self._soft_velocity_limits,
                                                                     self._soft_position_limits,
                                                                     acc_range_dynamic_vel[i],
                                                                     acc_range_dynamic_pos[i])

                    acc_range[i][0] = acc_range_min
                    acc_range[i][1] = acc_range_max
                    limit_violation[i] = limit_violation_joint


        if self._normalize_acc_range:
            acc_range_min_max = np.swapaxes(acc_range, 0, 1)
            acc_range_min_max = normalize_batch(acc_range_min_max, self._acc_limits_min_max)
            acc_range = np.swapaxes(acc_range_min_max, 0, 1)

        return acc_range, limit_violation



cdef (double, double, int) calculate_valid_acceleration_range_per_joint(int joint_index, double t_s, double current_pos, double current_vel, double current_acc,
                                                  double[:] pos_limits, double[:] vel_limits, double[:] acc_limits, double[:] jerk_limits,
                                                  double[:] acc_limits_after_max_vel,
                                                  bint set_velocity_after_max_pos_to_zero,
                                                  bint limit_velocity, bint limit_position,
                                                  bint braking_trajectory,
                                                  int time_step_counter,
                                                  double[:] limit_min_max, bint soft_velocity_limits, 
                                                  bint soft_position_limits,
                                                  double[:] acc_range_dynamic_vel,
                                                  double[:] acc_range_dynamic_pos) nogil:
    

    cdef int j, nj, min_max, limit_violation_code
    cdef double acc_range_jerk_min, acc_range_jerk_max
    cdef double a, b, c, n, a1_limit, t_a0_1, a_n_plus_1
    cdef double j_min, j_max, a_min, p_max, a_0, v_0, p_0  
    cdef double a_1_all_phases, a_1_reduced_jerk, a_1_min_jerk, a_1_min_first, a_1_upper_bound, a_1_bounded_vel_discrete_all_phases, a_1_bounded_vel_discrete_min_jerk
    cdef double t_v0_bounded_vel_min_jerk_phase, t_v0_min_jerk, t_v0_min_first, t_v0_upper_bound, t_v0_all_phases, t_v0_reduced_jerk
    cdef double t_u_bounded_vel_continuous_all_phases, t_u_bounded_vel_continuous_min_jerk, j_n_u_plus_1_all_phases, a_n_a_min, j_n_u_plus_1_min_jerk_phase
    cdef double t_a_min, t_star_all_phases, t_a_min_upper_bound, t_n_u_all_phases, t_n_u_min_jerk_phase, t_star_min_jerk_phase
    cdef double acc_range_min, acc_range_min_clipped, acc_range_max, acc_range_max_clipped
    cdef bint second_phase

    acc_range_jerk_min = current_acc + jerk_limits[0] * t_s
    acc_range_jerk_max = current_acc + jerk_limits[1] * t_s

    acc_range_dynamic_pos[0] = -1e6 
    acc_range_dynamic_pos[1] = 1e6

    if limit_velocity:
        if not braking_trajectory and (current_acc < 0 and (
                current_vel < vel_limits[0] + 0.5 * (current_acc ** 2 * t_s) / (acc_limits[1] - current_acc))):
            acc_range_dynamic_vel[0] = acc_limits[1]
        else:
            if not braking_trajectory and (current_acc > 0 and (
                    current_vel > vel_limits[1] - 0.5 * (current_acc ** 2 * t_s) / (
                    current_acc - acc_limits[0]))):
                acc_range_dynamic_vel[1] = acc_limits[0]
            else:
                
                for j in range(2):
                    if not limit_min_max[j]:
                        continue
                    nj = (j + 1) % 2
                    second_phase = True
                    if (j == 0 and (current_vel + 0.5 * current_acc * t_s) <= vel_limits[0]) \
                            or (j == 1 and (current_vel + 0.5 * current_acc * t_s) >= vel_limits[1]):

                        second_phase = False
                        if fabs(current_acc) < 1e-8:
                            acc_range_dynamic_vel[j] = 0
                        else:
                            if vel_limits[j] != current_vel:
                                acc_range_dynamic_vel[j] = current_acc * (
                                        1 - ((0.5 * current_acc * t_s) / (vel_limits[j] - current_vel)))
                            else:
                                acc_range_dynamic_vel[j] = nan

                        if braking_trajectory:
                            if fabs(acc_range_dynamic_vel[j]) >= 0.01 or isnan(acc_range_dynamic_vel[j]):
                                second_phase = True
                                nj = j

                    if second_phase:
                        a = - jerk_limits[nj] / 2
                        b = t_s * jerk_limits[nj] / 2
                        c = current_vel - vel_limits[j] + current_acc * t_s / 2

                        if b ** 2 - 4 * a * c >= 0:
                            if nj == 1:
                                t_a0_1 = (-b - sqrt(b ** 2 - 4 * a * c)) / (
                                        2 * a)
                            else:
                                t_a0_1 = (-b + sqrt(b ** 2 - 4 * a * c)) / (
                                        2 * a)

                            a1_limit = - jerk_limits[nj] * (t_a0_1 - t_s)

                            if ceil(t_a0_1 / t_s) > t_a0_1 / t_s:
                                n = ceil(t_a0_1 / t_s) - 1.0
                                a_n_plus_1 = a1_limit + jerk_limits[nj] * t_s * n
                                if (nj == 1 and a_n_plus_1 > acc_limits_after_max_vel[1]) or \
                                        (nj == 0 and a_n_plus_1 < acc_limits_after_max_vel[0]):
                    
                                    if j == nj:
                                        min_max = (j + 1) % 2
                                    else:
                                        min_max = j

                                    a1_limit = \
                                        velocity_reduced_acceleration(min_max, jerk_limits[nj], current_acc, acc_limits_after_max_vel[nj],
                                                                      current_vel, vel_limits[j], t_s, n * t_s)

                            acc_range_dynamic_vel[j] = a1_limit
                        else:
                            pass
                    
                    if isnan(acc_range_dynamic_vel[j]):
                        acc_range_dynamic_vel[j] = acc_limits[j]

    if limit_position and (isnan(pos_limits[0]) or isnan(pos_limits[1])):
        # continuous joint -> no position limitation required
        limit_position = False

    if limit_position:
        for j in range(2):
            if not limit_min_max[j]:
                continue
            nj = (j + 1) % 2
            a_min = acc_limits[nj]
            j_min = jerk_limits[nj]
            j_max = jerk_limits[j]
            p_max = pos_limits[j]
            p_0 = current_pos
            v_0 = current_vel
            a_0 = current_acc
            a_1_all_phases = 0
            a_1_reduced_jerk = nan
            t_v0_bounded_vel_min_jerk_phase = nan
            t_star_all_phases = nan

            a_1_min_jerk, t_v0_min_jerk = position_border_case_min_jerk_phase(j_min, a_0, v_0, p_0, p_max, t_s)

            if t_v0_min_jerk < t_s + 1e-8 or isnan(t_v0_min_jerk):
                a_1_min_first, t_v0_min_first = position_border_case_first_phase(j, a_0, v_0, p_0, p_max, t_s)

                if 0 < t_v0_min_first <= t_s + 1e-3:
                    acc_range_dynamic_pos[j] = a_1_min_first
                else:

                    if p_0 == p_max and a_0 == 0 and v_0 == 0:
                        acc_range_dynamic_pos[j] = 0
            else:

                t_n_a_min = t_s * (1 + floor((a_min - a_1_min_jerk) / (j_min * t_s)))

                if t_n_a_min >= t_v0_min_jerk:
                    if t_v0_min_jerk >= t_s:
                        acc_range_dynamic_pos[j] = a_1_min_jerk

                        if set_velocity_after_max_pos_to_zero:
                            t_v0_bounded_vel_min_jerk_phase = t_v0_min_jerk

                else:
                    t_a_min = t_s * (1 + ((a_min - a_1_min_jerk) / (j_min * t_s)))

                    if t_v0_min_jerk > t_a_min:
                        a_1_upper_bound, t_v0_upper_bound = \
                            position_border_case_upper_bound(j, j_min, a_0, a_min, v_0, p_0, p_max, t_s)
                        if isnan(a_1_upper_bound):
                            acc_range_dynamic_pos[j] = a_0 + j_min
                            continue

                        t_a_min_upper_bound = t_s * (1 + ((a_min - a_1_upper_bound) / (j_min * t_s)))
                        if t_a_min_upper_bound < t_s:
                            if t_a_min_upper_bound / t_s > 0.999:
                                t_a_min_upper_bound = t_s

                        t_n_a_min = t_s * floor(t_a_min_upper_bound / t_s)

                    a_1_all_phases, t_v0_all_phases = \
                        position_border_case_all_phases(j, j_min, a_0, a_min, v_0, p_0, p_max, t_s, t_n_a_min)

                    if t_v0_all_phases >= t_n_a_min + t_s:
                        acc_range_dynamic_pos[j] = a_1_all_phases

                        if set_velocity_after_max_pos_to_zero:
                            t_star_all_phases = t_s * ceil(t_v0_all_phases / t_s)
                            _, t_u_bounded_vel_continuous_all_phases = \
                                position_bounded_velocity_continuous_all_phases(j_min, j_max, a_0, a_min, v_0,
                                                                                p_0, p_max, t_s,
                                                                                t_star_all_phases, t_n_a_min)

                            if t_u_bounded_vel_continuous_all_phases >= t_star_all_phases:
                                pass

                            if t_u_bounded_vel_continuous_all_phases >= t_n_a_min + t_s:
                                t_n_u_all_phases = t_s * floor(t_u_bounded_vel_continuous_all_phases / t_s)
                                a_1_bounded_vel_discrete_all_phases, j_n_u_plus_1_all_phases = \
                                    position_bounded_velocity_discrete_all_phases(j_min, j_max, a_0, a_min, v_0,
                                                                                  p_0, p_max, t_s,
                                                                                  t_star_all_phases, t_n_a_min,
                                                                                  t_n_u_all_phases)

                                a_n_a_min = a_1_bounded_vel_discrete_all_phases + (t_n_a_min - t_s) * j_min
                                if (j == 0 and a_n_a_min > a_min + 1e-3) or \
                                        (j == 1 and a_n_a_min < a_min - 1e-3):

                                    if round(t_n_a_min / t_s) > 1:
                                        a_1_bounded_vel_discrete_all_phases, j_n_u_plus_1_all_phases = \
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

                        a_1_reduced_jerk, t_v0_reduced_jerk = position_border_case_reduced_jerk_phase(j_min, a_0, a_min, 
                                                                                                      v_0, p_0, p_max, t_s,
                                                                                                      t_n_a_min)

                        acc_range_dynamic_pos[j] = a_1_reduced_jerk

                if set_velocity_after_max_pos_to_zero and not isnan(t_v0_bounded_vel_min_jerk_phase):
                    t_star_min_jerk_phase = t_s * ceil(t_v0_bounded_vel_min_jerk_phase / t_s)

                    if t_star_min_jerk_phase >= 3 * t_s:
                        _, t_u_bounded_vel_continuous_min_jerk = \
                            position_bounded_velocity_continuous_min_jerk_phase(j_min, j_max, a_0, v_0, p_0,
                                                                                p_max, t_s,
                                                                                t_star_min_jerk_phase)

                        if not isnan(t_u_bounded_vel_continuous_min_jerk) and \
                                (t_u_bounded_vel_continuous_min_jerk / t_s) > 0.99:
                            t_n_u_min_jerk_phase = fmax(t_s * floor(t_u_bounded_vel_continuous_min_jerk / t_s),
                                                        t_s)
                        else:
                            t_n_u_min_jerk_phase = nan

                    else:
                        t_n_u_min_jerk_phase = t_s

                    if not isnan(t_n_u_min_jerk_phase):
                        a_1_bounded_vel_discrete_min_jerk, j_n_u_plus_1_min_jerk_phase = \
                            position_bounded_velocity_discrete_min_jerk_phase(j_min, j_max, a_0, v_0, p_0,
                                                                              p_max, t_s,
                                                                              t_star_min_jerk_phase,
                                                                              t_n_u_min_jerk_phase)

                        if (j == 0 and j_max - 1e-6 <= j_n_u_plus_1_min_jerk_phase <= j_min + 1e-6) or \
                                (j == 1 and j_min - 1e-6 <= j_n_u_plus_1_min_jerk_phase <= j_max + 1e-6):

                            if not isnan(t_star_all_phases):
                                if (j == 0 and a_1_bounded_vel_discrete_min_jerk > a_1_all_phases) or \
                                        (j == 1 and a_1_bounded_vel_discrete_min_jerk < a_1_all_phases):

                                    acc_range_dynamic_pos[j] = a_1_bounded_vel_discrete_min_jerk
                                else:
                                    pass
                            else:
                                if not isnan(a_1_reduced_jerk):
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

            if isnan(acc_range_dynamic_pos[j]):
                acc_range_dynamic_pos[j] = acc_limits[j]

    if limit_velocity:
        if soft_velocity_limits:
            acc_range_dynamic_vel[0] = fmax(acc_range_jerk_min, acc_range_dynamic_vel[0])
            acc_range_dynamic_vel[1] = fmin(acc_range_jerk_max, acc_range_dynamic_vel[1])

            if acc_range_dynamic_vel[1] < acc_range_jerk_min:
                acc_range_dynamic_vel[1] = acc_range_jerk_min
            if acc_range_dynamic_vel[0] > acc_range_jerk_max:
                acc_range_dynamic_vel[0] = acc_range_jerk_max

    if limit_position:
        if soft_position_limits and soft_velocity_limits:
            acc_range_dynamic_pos[0] = fmax(acc_range_jerk_min, acc_range_dynamic_pos[0])
            acc_range_dynamic_pos[1] = fmin(acc_range_jerk_max, acc_range_dynamic_pos[1])

            if acc_range_dynamic_pos[1] < acc_range_jerk_min:
                acc_range_dynamic_pos[1] = acc_range_jerk_min
            if acc_range_dynamic_pos[0] > acc_range_jerk_max:
                acc_range_dynamic_pos[0] = acc_range_jerk_max

    limit_violation_code = 0

    acc_range_min = acc_range_jerk_min
    acc_range_max = acc_range_jerk_max

    if limit_velocity:
        acc_range_min = fmax(acc_range_min, acc_range_dynamic_vel[0])
        acc_range_max = fmin(acc_range_max, acc_range_dynamic_vel[1])
    if limit_position:
        acc_range_min = fmax(acc_range_min, acc_range_dynamic_pos[0])
        acc_range_max = fmin(acc_range_max, acc_range_dynamic_pos[1])

    if (acc_range_min - acc_range_max) > 0.001:
        limit_violation_code = limit_violation_code + 1
        if limit_position and limit_velocity:
            acc_range_min = fmax(acc_range_dynamic_vel[0], acc_range_dynamic_pos[0])
            acc_range_max = fmin(acc_range_dynamic_vel[1], acc_range_dynamic_pos[1])
            if (acc_range_min - acc_range_max) > 0.001:
                limit_violation_code = limit_violation_code + 1
                acc_range_min = acc_range_dynamic_pos[0]
                acc_range_max = acc_range_dynamic_pos[1]
        else:
            if limit_velocity:
                acc_range_min = acc_range_dynamic_vel[0]
                acc_range_max = acc_range_dynamic_vel[1]
            else:
                acc_range_min = acc_range_dynamic_pos[0]
                acc_range_max = acc_range_dynamic_pos[1]
  
    if acc_range_min > acc_range_max:
        acc_range_max = acc_range_min
        
    acc_range_min_clipped = fmin(acc_limits[1], fmax(acc_range_min, acc_limits[0]))
    acc_range_max_clipped = fmin(acc_limits[1], fmax(acc_range_max, acc_limits[0]))

    if (acc_range_min_clipped != acc_range_min) or (acc_range_max_clipped != acc_range_max):
        limit_violation_code = limit_violation_code + 1

    return acc_range_min_clipped, acc_range_max_clipped, limit_violation_code


cdef double velocity_reduced_acceleration(int min_max, double j_min_in, double a_0_in, double a_n_plus_1_star_in, 
                                          double v_0_in, double v_max_in, double t_s_in, double t_n_in) nogil:

    cdef double a_1_out
    if a_n_plus_1_star_in == 0:
        a_1_out = vel_zero_a_n_plus_1_star_a1(j_min_in, a_0_in, v_0_in, v_max_in, t_s_in, t_n_in)
    else:
        if min_max == 0:
            a_1_out = vel_fixed_a_n_plus_1_star_a1_min(j_min_in, a_0_in, a_n_plus_1_star_in, v_0_in,
                                                       v_max_in, t_s_in, t_n_in)
        else:
            a_1_out = vel_fixed_a_n_plus_1_star_a1_max(j_min_in, a_0_in, a_n_plus_1_star_in, v_0_in,
                                                       v_max_in, t_s_in, t_n_in)

    return a_1_out

cdef (double, double) position_border_case_min_jerk_phase(double j_min_in, double a_0_in, double v_0_in, double p_0_in, double p_max_in, double t_s_in) nogil:

    cdef double complex t_v0_out_complex
    cdef double t_v0_out
    cdef double a_1_out

    t_v0_out_complex = pos_min_jerk_tv0_2(j_min_in, a_0_in, v_0_in, p_0_in, p_max_in, t_s_in)

    if fabs(t_v0_out_complex.imag) < 1e-5:
        t_v0_out = t_v0_out_complex.real
    else:
        t_v0_out = nan

    a_1_out = pos_min_jerk_a1(j_min_in, a_0_in, v_0_in, t_s_in, t_v0_out)

    return a_1_out, t_v0_out

cdef (double, double) position_border_case_first_phase(int min_max, double a_0_in, double v_0_in, double p_0_in, double p_max_in, double t_s_in) nogil:
    
    cdef double t_v0_out
    cdef double a_1_out

    if min_max == 0:
        t_v0_out = pos_first_tv0_min(a_0_in, v_0_in, p_0_in, p_max_in)
    else:
        t_v0_out = pos_first_tv0_max(a_0_in, v_0_in, p_0_in, p_max_in)

    a_1_out = pos_first_a1(a_0_in, v_0_in, t_s_in, t_v0_out)

    return a_1_out, t_v0_out

cdef (double, double) position_border_case_upper_bound(int min_max, double j_min_in, double a_0_in, double a_min_in, double v_0_in, double p_0_in, double p_max_in, double t_s_in) nogil:

    cdef bint compute_t_v0 = False

    cdef double t_v0_out
    cdef double complex a_1_out_complex
    cdef double a_1_out

    if min_max == 0:
        a_1_out_complex = pos_upper_bound_a1_min_2(j_min_in, a_0_in, a_min_in, v_0_in, p_0_in, p_max_in, t_s_in)
    else:
        a_1_out_complex = pos_upper_bound_a1_max_2(j_min_in, a_0_in, a_min_in, v_0_in, p_0_in, p_max_in, t_s_in)
                
    if fabs(a_1_out_complex.imag) < 1e-3:
        a_1_out = a_1_out_complex.real
    else:
        if min_max == 0:
            a_1_out_complex = pos_upper_bound_a1_min_5(j_min_in, a_0_in, a_min_in, v_0_in, p_0_in, p_max_in, t_s_in)
        else:
            a_1_out_complex = pos_upper_bound_a1_max_5(j_min_in, a_0_in, a_min_in, v_0_in, p_0_in, p_max_in, t_s_in)
                
        if fabs(a_1_out_complex.imag) < 1e-3:
            a_1_out = a_1_out_complex.real
        else:
            if min_max == 0:
                a_min_in = a_min_in - 0.02
            else:
                a_min_in = a_min_in + 0.02
            if min_max == 0:
                a_1_out_complex = pos_upper_bound_a1_min_2(j_min_in, a_0_in, a_min_in, v_0_in, p_0_in, p_max_in, t_s_in)
            else:
                a_1_out_complex = pos_upper_bound_a1_max_2(j_min_in, a_0_in, a_min_in, v_0_in, p_0_in, p_max_in, t_s_in)
                
            if fabs(a_1_out_complex.imag) < 1e-3:
                a_1_out = a_1_out_complex.real
            else:
                if min_max == 0:
                    a_1_out_complex = pos_upper_bound_a1_min_5(j_min_in, a_0_in, a_min_in, v_0_in, p_0_in, p_max_in, t_s_in)
                else:
                    a_1_out_complex = pos_upper_bound_a1_max_5(j_min_in, a_0_in, a_min_in, v_0_in, p_0_in, p_max_in, t_s_in)
                        
                if fabs(a_1_out_complex.imag) < 1e-3:
                    a_1_out = a_1_out_complex.real
                else:
                    a_1_out = nan

    if compute_t_v0:
        t_v0_out = pos_upper_bound_tv0(j_min_in, a_0_in, a_1_out, a_min_in, v_0_in, t_s_in)
    else:
        t_v0_out = nan

    return a_1_out, t_v0_out

cdef (double, double) position_border_case_all_phases(int min_max, double j_min_in, double a_0_in, double a_min_in, double v_0_in, double p_0_in, 
                                     double p_max_in, double t_s_in, double t_n_a_min_in) nogil:

    cdef double t_v0_out
    cdef double a_1_out
    
    if min_max == 0:
        a_1_out = pos_all_a1_min(j_min_in, a_0_in, a_min_in, v_0_in, p_0_in, p_max_in, t_s_in, t_n_a_min_in)
    else:
        a_1_out = pos_all_a1_max(j_min_in, a_0_in, a_min_in, v_0_in, p_0_in, p_max_in, t_s_in, t_n_a_min_in)

    t_v0_out = pos_all_tv0(j_min_in, a_0_in, a_1_out, a_min_in, v_0_in, t_s_in, t_n_a_min_in)

    return a_1_out, t_v0_out

cdef (double, double) position_border_case_reduced_jerk_phase(double j_min_in, double a_0_in, double a_min_in, double v_0_in, double p_0_in, double p_max_in, 
                                             double t_s_in, double t_n_a_min_in) nogil:

    cdef double complex t_v0_out_complex
    cdef double t_v0_out
    cdef double a_1_out

    t_v0_out_complex = pos_reduced_jerk_tv0_2(j_min_in, a_0_in, a_min_in, v_0_in, p_0_in, p_max_in, t_s_in, t_n_a_min_in)

    if fabs(t_v0_out_complex.imag) < 1e-5 and t_v0_out_complex.real >= t_n_a_min_in:
        t_v0_out = t_v0_out_complex.real
    else:
        t_v0_out_complex = pos_reduced_jerk_tv0_5(j_min_in, a_0_in, a_min_in, v_0_in, p_0_in, p_max_in, t_s_in, t_n_a_min_in)

        if fabs(t_v0_out_complex.imag) < 1e-5 and t_v0_out_complex.real >= t_n_a_min_in:
            t_v0_out = t_v0_out_complex.real
        else:
            t_v0_out = nan
        
    a_1_out = pos_reduced_jerk_a1(j_min_in, a_0_in, a_min_in, v_0_in, t_s_in, t_v0_out, t_n_a_min_in)

    return a_1_out, t_v0_out

cdef (double, double) position_bounded_velocity_continuous_all_phases(double j_min_in, double j_max_in, double a_0_in, double a_min_in, double v_0_in, double p_0_in,
                                                     double p_max_in, double t_s_in, double t_star_in, double t_n_a_min_in) nogil:

    cdef bint compute_a_1 = False

    cdef double t_u_out
    cdef double complex t_u_out_complex
    cdef double a_1_out

    t_u_out_complex = pos_all_bounded_vel_continuous_tu(j_min_in, j_max_in, a_0_in, a_min_in,
                                                        v_0_in, p_0_in, p_max_in, t_s_in, 
                                                        t_star_in, t_n_a_min_in)
    if fabs(t_u_out_complex.imag) < 1e-8:
        t_u_out = t_u_out_complex.real
    else:
        t_u_out = nan

    if compute_a_1:
        a_1_out = pos_all_bounded_vel_continuous_a1(j_min_in, j_max_in, a_0_in, a_min_in, v_0_in, 
                                                    t_s_in, t_star_in, t_n_a_min_in, t_u_out)
    else:
        a_1_out = nan

    return a_1_out, t_u_out

cdef (double, double) position_bounded_velocity_discrete_all_phases(double j_min_in, double j_max_in, double a_0_in, double a_min_in, double v_0_in, 
                                                   double p_0_in, double p_max_in, double t_s_in, double t_star_in, 
                                                   double t_n_a_min_in, double t_n_u_in) nogil:

    cdef double j_n_u_plus_1_out
    cdef double complex j_n_u_plus_1_out_complex
    cdef double a_1_out
    
    j_n_u_plus_1_out_complex = pos_all_bounded_vel_discrete_j_n_u_plus_1(j_min_in, j_max_in, a_0_in, a_min_in, v_0_in, p_0_in,
                                                                         p_max_in, t_s_in, t_star_in, t_n_a_min_in, t_n_u_in)

    if fabs(j_n_u_plus_1_out_complex.imag) < 1e-8:
        j_n_u_plus_1_out = j_n_u_plus_1_out_complex.real
    else:
        j_n_u_plus_1_out = nan

    a_1_out = pos_all_bounded_vel_discrete_a1(j_min_in, j_max_in, j_n_u_plus_1_out, a_0_in, a_min_in, v_0_in, t_s_in, 
                                              t_star_in, t_n_a_min_in, t_n_u_in)

    return a_1_out, j_n_u_plus_1_out

cdef (double, double) position_bounded_velocity_continuous_min_jerk_phase(double j_min_in, double j_max_in, double a_0_in, double v_0_in, double p_0_in, 
                                                         double p_max_in, double t_s_in, double t_star_in) nogil:

    cdef bint compute_a_1 = False

    cdef double t_u_out
    cdef double complex t_u_out_complex
    cdef double a_1_out
    
    t_u_out_complex = pos_min_jerk_bounded_vel_continuous_tu(j_min_in, j_max_in, a_0_in, v_0_in, p_0_in, 
                                                             p_max_in, t_s_in, t_star_in)

    if fabs(t_u_out_complex.imag) < 1e-8:
        t_u_out = t_u_out_complex.real
    else:
        t_u_out = nan

    if compute_a_1:
        a_1_out = pos_min_jerk_bounded_vel_continuous_a1(j_min_in, j_max_in, a_0_in, v_0_in, t_s_in,
                                                         t_star_in, t_u_out)
    else:
        a_1_out = nan

    return a_1_out, t_u_out

cdef (double, double) position_bounded_velocity_discrete_min_jerk_phase(double j_min_in, double j_max_in, double a_0_in, double v_0_in, double p_0_in, 
                                                       double p_max_in, double t_s_in, double t_star_in, double t_n_u_in) nogil:

    cdef double j_n_u_plus_1_out
    cdef double complex j_n_u_plus_1_out_complex
    cdef double a_1_out
    
    j_n_u_plus_1_out_complex = pos_min_jerk_bounded_vel_discrete_j_n_u_plus_1(j_min_in, j_max_in, a_0_in, 
                                                                              v_0_in, p_0_in, p_max_in, 
                                                                              t_s_in, t_star_in, t_n_u_in)

    if fabs(j_n_u_plus_1_out_complex.imag) < 1e-8:
        j_n_u_plus_1_out = j_n_u_plus_1_out_complex.real
    else:
        j_n_u_plus_1_out = nan

    a_1_out = pos_min_jerk_bounded_vel_discrete_a1(j_min_in, j_max_in, j_n_u_plus_1_out, a_0_in, 
                                                   v_0_in, t_s_in, t_star_in, t_n_u_in)

    return a_1_out, j_n_u_plus_1_out