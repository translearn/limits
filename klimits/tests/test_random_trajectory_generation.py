import pytest

from klimits.test_trajectory_generation import test_trajectory_generation


@pytest.mark.parametrize("time_step", [0.05, 0.1])
@pytest.mark.parametrize("pos_limits", [[[-2.96705972839, 2.96705972839],
                                         [-2.09439510239, 2.09439510239],
                                         [-2.96705972839, 2.96705972839],
                                         [-2.09439510239, 2.09439510239],
                                         [-2.96705972839, 2.96705972839],
                                         [-2.09439510239, 2.09439510239],
                                         [-3.05432619099, 3.05432619099]]])
@pytest.mark.parametrize("vel_limits", [[[-1.71042266695, 1.71042266695],
                                         [-1.71042266695, 1.71042266695],
                                         [-1.74532925199, 1.74532925199],
                                         [-2.26892802759, 2.26892802759],
                                         [-2.44346095279, 2.44346095279],
                                         [-3.14159265359, 3.14159265359],
                                         [-3.14159265359, 3.14159265359]]])
@pytest.mark.parametrize("acc_limits", [[[-15.0, 15.0],
                                         [-7.5, 7.5],
                                         [-10.0, 10.0],
                                         [-12.5, 12.5],
                                         [-15.0, 15.0],
                                         [-20.0, 20.0],
                                         [-20.0, 20.0]]])
@pytest.mark.parametrize("pos_limit_factor", [0.2, 0.5, 1.0])
@pytest.mark.parametrize("vel_limit_factor", [0.2, 0.5, 1.0])
@pytest.mark.parametrize("acc_limit_factor", [0.2, 0.5, 1.0])
@pytest.mark.parametrize("jerk_limit_factor", [0.2, 0.5, 1.0])
@pytest.mark.parametrize("trajectory_duration", [300])
@pytest.mark.parametrize("seed", [1])
def test_random_trajectory_generation(time_step, pos_limits, vel_limits, acc_limits, pos_limit_factor,
                                      vel_limit_factor, acc_limit_factor, jerk_limit_factor, trajectory_duration,
                                      seed):
    trajectory_summary = test_trajectory_generation(time_step=time_step, pos_limits=pos_limits, vel_limits=vel_limits,
                                                    acc_limits=acc_limits, pos_limit_factor=pos_limit_factor,
                                                    vel_limit_factor=vel_limit_factor,
                                                    acc_limit_factor=acc_limit_factor,
                                                    jerk_limit_factor=jerk_limit_factor,
                                                    trajectory_duration=trajectory_duration,
                                                    constant_action=None, num_threads=1, plot_joint=None,
                                                    no_plot=True, plot_safe_acc_limits=False, seed=seed,
                                                    return_summary=True)

    for key, value in trajectory_summary.items():
        for joint_index in range(len(value)):
            assert -1.001 < value[joint_index]['min'], \
                "min {} violation, joint {}, value {}".format(key, joint_index, value[joint_index]['min'])
            assert value[joint_index]['max'] < 1.001, \
                "max {} violation, joint {}, value {}".format(key, joint_index, value[joint_index]['max'])
