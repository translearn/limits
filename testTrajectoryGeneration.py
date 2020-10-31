import math
import numpy as np
import posVelJerkLimits
import trajectoryPlotter
import timeit

'''
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
'''


def denormalize(normValue, valueRange):
    actualValue = valueRange[0] + 0.5 * (normValue + 1) * (valueRange[1] - valueRange[0])
    return actualValue


if __name__ == '__main__':
    #  beginning of user settings -------------------------------------------------------------------------------

    # time between network predictions -> timeStep = 1 / predictionFrequency
    timeStep = 1 / 20   # 50 ms

    # factors to limit the maximum position, velocity, acceleration and jerk
    # (relative to the actual limits specified below)
    posLimitFactor = 1  # <= 1.0
    velLimitFactor = 1  # <= 1.0
    accLimitFactor = 1  # <= 1.0
    jerkLimitFactor = 1  # <= 1.0

    trajectoryDuration = 10  # duration of the generated trajectory in seconds

    # True: Trajectory for the corresponding joint is plotted
    plotJoint = [True, True, True, False, False, False, False]
    plotAccLimits = False  # True: The calculated range of safe accelerations is plotted with dashed lines

    useRandomActions = True
    # if True: actions to generate the trajectory are randomly sampled
    # if False: the constant action stored in constantAction is used at each decision step
    constantAction = 0.96  # scalar within [-1, 1]

    useMappingStrategy = True  # True: Mapping, False: Clipping

    posLimits = [[-2.96705972839, 2.96705972839],  # min, max Joint 1 in rad
                 [-2.09439510239, 2.09439510239],
                 [-2.96705972839, 2.96705972839],
                 [-2.09439510239, 2.09439510239],
                 [-2.96705972839, 2.96705972839],
                 [-2.09439510239, 2.09439510239],
                 [-3.05432619099, 3.05432619099]]  # min, max Joint 7 in rad

    velLimits = [[-1.71042266695, 1.71042266695],  # min, max Joint 1 in rad/s
                 [-1.71042266695, 1.71042266695],
                 [-1.74532925199, 1.74532925199],
                 [-2.26892802759, 2.26892802759],
                 [-2.44346095279, 2.44346095279],
                 [-3.14159265359, 3.14159265359],
                 [-3.14159265359, 3.14159265359]]  # min, max Joint 7 in rad/s

    accLimits = [[-15, 15],  # min, max Joint 1 in rad/s^2
                 [-7.5, 7.5],
                 [-10, 10],
                 [-12.5, 12.5],
                 [-15, 15],
                 [-20, 20],
                 [-20, 20]]  # min, max Joint 7 in rad/s^2

    #  end of user settings -------------------------------------------------------------------------------

    accLimits = [[accLimitFactor * accLimit[0], accLimitFactor * accLimit[1]] for accLimit in accLimits]
    maxJerks = [(accLimit[1] - accLimit[0]) / timeStep for accLimit in accLimits]
    jerkLimits = [[-jerkLimitFactor * maxJerk, jerkLimitFactor * maxJerk] for maxJerk in maxJerks]
    velLimits = [[velLimitFactor * velLimit[0], velLimitFactor * velLimit[1]] for velLimit in velLimits]
    posLimits = [[posLimitFactor * posLimit[0], posLimitFactor * posLimit[1]] for posLimit in posLimits]

    accLimitation = posVelJerkLimits.PosVelJerkLimitation(timeStep=timeStep,
                                                          posLimits=posLimits, velLimits=velLimits,
                                                          accLimits=accLimits, jerkLimits=jerkLimits,
                                                          accelerationAfterMaxVelLimitFactor=0.0001,
                                                          setVelocityAfterMaxPosToZero=True,
                                                          limitVelocity=True, limitPosition=True, numWorkers=1)

    trajectoryPlotter = trajectoryPlotter.TrajectoryPlotter(timeStep=timeStep,
                                                            posLimits=posLimits, velLimits=velLimits,
                                                            accLimits=accLimits,
                                                            jerkLimits=jerkLimits,
                                                            plotJoint=plotJoint,
                                                            plotAccLimits=plotAccLimits)

    currentPosition = [0 for jointRange in posLimits]
    currentVelocity = [0 for jointRange in velLimits]
    currentAcceleration = [0 for jointRange in accLimits]

    trajectoryPlotter.resetPlotter(currentPosition)

    trajectoryTimer = timeit.default_timer()
    print('Calculating trajectory ...')

    for j in range(round(trajectoryDuration / timeStep)):

        # calculate the range of valid actions
        safeActionRange, _ = accLimitation.calculateValidAccelerationRange(currentPosition,
                                                                           currentVelocity,
                                                                           currentAcceleration, j)

        # generate actions in range [-1, 1] for each joint
        # Note: Action calculation is normally performed by a neural network
        if useRandomActions:
            action = [np.random.uniform(low=-1, high=1) for jointRange in posLimits]
        else:
            action = [constantAction for jointRange in posLimits]

        # clip or map the chosen action to the range of safe accelerations
        if useMappingStrategy:
            safeAction = np.array([safeActionRange[i][0] + 0.5 * (action[i] + 1) *
                                   (safeActionRange[i][1] - safeActionRange[i][0])
                                   for i in range(len(action))])
        else:
            safeAction = np.array([np.clip(action[i], safeActionRange[i][0], safeActionRange[i][1])
                                   for i in range(len(action))])

        trajectoryPlotter.addDataPoint(safeAction, safeActionRange)

        nextAcceleration = [denormalize(safeAction[k], accLimits[k]) for k in range(len(safeAction))]

        nextPosition = [currentPosition[k] + currentVelocity[k] * timeStep +
                        (1 / 3 * currentAcceleration[k] + 1 / 6 * nextAcceleration[k]) * timeStep ** 2
                        for k in range(len(currentPosition))]

        nextVelocity = [currentVelocity[k] + 0.5 * timeStep * (currentAcceleration[k] + nextAcceleration[k])
                        for k in range(len(currentVelocity))]

        currentPosition = nextPosition
        currentVelocity = nextVelocity
        currentAcceleration = nextAcceleration

    print('Calculating and plotting a trajectory of ' + str(trajectoryDuration) + " seconds took " +
          str(timeit.default_timer() - trajectoryTimer) + ' seconds')
    trajectoryPlotter.displayPlot()
