# Learning Robot Trajectories subject to Kinematic Joint Constraints
[![ICRA 2021](https://img.shields.io/badge/ICRA-2021-%3C%3E)](https://www.ieee-icra.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2011.00563-b31b1b)](https://arxiv.org/abs/2011.00563)
[![PyPI version](https://img.shields.io/pypi/v/klimits)](https://pypi.python.org/pypi/klimits/)
[![PyPI license](https://img.shields.io/pypi/l/klimits)](https://pypi.python.org/pypi/klimits)
[![GitHub issues](https://img.shields.io/github/issues/translearn/limits)](https://github.com/translearn/limits/issues/)
[![PyPI download month](https://img.shields.io/pypi/dm/klimits)](https://pypi.python.org/pypi/klimits/) <br>

This python package enables learning of robot trajectories without exceeding limits on the position, velocity, acceleration and jerk of each robot joint.
Movements are generated by mapping the predictions of a neural network to safely executable joint accelerations. The time between network predictions must be constant, but can be chosen arbitrarily.
Our method ensures that the kinematic constraints are never in conflict, which means that there is at least one safely executable joint acceleration at any time. \
This package provides the code to compute the range of safely executable joint accelerations.

## Installation

The package can be installed by running

    pip install klimits

## Trajectory generation &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/translearn/notebooks/blob/main/klimits_demo.ipynb)

To generate a random trajectory with limited jerk, acceleration, velocity and position run

    python -m klimits.test_trajectory_generation

Several parameters can be adjusted to modify the generated trajectory. E.g:

    python -m klimits.test_trajectory_generation --time_step=0.1 --pos_limits='[[-2.96705972839, 2.96705972839], [-2.09439510239, 2.09439510239]]' --vel_limits='[[-1.71042266695, 1.71042266695], [-1.71042266695, 1.71042266695]]' --acc_limits='[[-15, 15], [-7.5, 7.5]]' --plot_joint='[1, 0]' --pos_limit_factor=0.9 --vel_limit_factor=0.8 --acc_limit_factor=0.7 --jerk_limit_factor=0.6 --trajectory_duration=20 --plot_safe_acc_limits 
Run

    python -m klimits.test_trajectory_generation --help

for further details on the optional arguments.

<p align="center">
<img src="https://user-images.githubusercontent.com/51738372/116689339-43b5da00-a9b8-11eb-9775-193dec48e00f.png" width=70% height=70% alt="exemplary_trajectory">
</p>


## Further reading

A preprint of the corresponding ICRA 2021 publication is available at [arXiv.org](https://arxiv.org/abs/2011.00563). \
Further information on the implementation can be found [here](https://www.researchgate.net/publication/350451653_Background_Knowledge_for_Learning_Robot_Trajectories_subject_to_Kinematic_Joint_Constraints). \
This library is used by [safeMotions](https://github.com/translearn/safemotions) to learn collision-free reaching tasks via reinforcement learning.

## Video

<p align="center">
<a href="https://www.youtube.com/watch?v=pzXOxE7y7ws">
<img src="https://user-images.githubusercontent.com/51738372/131867526-15270aee-a1dc-47fc-a4df-54b6d07fbcc5.png" alt="video">
</a>
</p>

## Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
