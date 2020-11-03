# Learning Robot Trajectories subject to Kinematic Joint Constraints

Trajectories with limited jerk, acceleration, velocity and position can be generated by running

    python testTrajectoryGeneration.py

Once the code is executed for the first time, time-consuming parts of the code are automatically compiled as python modules and stored in a newly generated build directory.
SymPy >= 1.5.1, Cython and gcc are required for this process. 
The compilation process might take a few minutes. 

Several parameters can be adjusted to modify the generated trajectories.
See the comments provided in [testTrajectoryGeneration.py](testTrajectoryGeneration.py) fur further details.


Our code was tested using Ubuntu 16.04. and a virtual python envinronment including the following packages:

    python 3.7.3
    cython 0.29.12
    numpy  1.16.4
    sympy  1.5.1
    matplotlib 3.1.0

A preprint of the corresponding publication is available at [arXiv.org](https://arxiv.org/abs/2011.00563). \
Further information on the implementation can be found in [docs/background_knowledge.pdf](docs/background_knowledge.pdf).
