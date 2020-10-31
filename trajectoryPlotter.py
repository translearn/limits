import numpy as np
import os
import inspect
import matplotlib
import matplotlib.pyplot as plt  
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import datetime
import json
import errno

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

class TrajectoryPlotter():
    def __init__(self,
                 timeStep=None,
                 controlTimeStep=None,
                 posLimits=None,
                 velLimits=None,
                 accLimits=None,
                 jerkLimits=None,
                 plotJoint=None,
                 plotAccLimits=False,
                 plotForPaper=0,
                 plotForVideo=0,
                 plotTimeLimits=None,
                 plotActualValues=False):

        self._timeStep = timeStep
        self._controlTimeStep = controlTimeStep
        self._plotForPaper = plotForPaper
        self._plotForVideo = plotForVideo
        self._plotAccLimits = plotAccLimits
        self._plotTimeLimits = plotTimeLimits  
        self._plotActualValues = plotActualValues

        if timeStep:
            self._plotNumSubTimeSteps = int(1000 * timeStep)

            self._timeStepCounter = None

            
            
            self._trajectoryCounter = 0


            self._currentJerk = None
            self._currentAcc = None
            self._currentVel = None
            self._currentPos = None

            self._time = None
            self._pos = None
            self._vel = None
            self._acc = None
            self._jerk = None

            self._subTime = None
            self._subPos = None
            self._subVel = None
            self._subAcc = None
            self._subJerk = None

            self._posLimits = posLimits
            self._velLimits = velLimits
            self._accLimits = accLimits
            self._jerkLimits = jerkLimits


            self._numJoints = len(self._posLimits)

            if plotJoint is None:
                self._plotJoint = [True for i in range(self._numJoints)]
            else:
                self._plotJoint = plotJoint


            self._episodeCounter = 0



            self._zero_vector = [0 for i in range(self._numJoints)]

            self._currentAccLimits = None
            self._actualPos = None

            self._timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        matplotlib.rcParams['text.usetex'] = True

        if self._plotForPaper != 0:
            font = {'family': 'normal',
                    'weight': 'normal',
                    'size': 24}

            matplotlib.rc('font', **font)

        if self._plotForVideo != 0:
            font = {'family': 'normal',
                    'weight': 'normal',
                    'size': 16}

            matplotlib.rc('font', **font)

    @property
    def trajectory_time(self):
        return self._time[-1]

    def resetPlotter(self, initial_joint_position):
        

        self._timeStepCounter = 0
        self._trajectoryCounter = self._trajectoryCounter + 1  

        self._episodeCounter = self._episodeCounter + 1

        self._currentAcc = self._zero_vector.copy()
        self._currentVel = self._zero_vector.copy()
        self._currentPos = initial_joint_position.copy()

        self._pos = []
        self._vel = []
        self._acc = []
        self._jerk = []
        self._currentAccLimits = []
        self._currentAccLimits.append([[0, 0] for i in range(self._numJoints)])

        self._subPos = []
        self._subVel = []
        self._subAcc = []
        self._subJerk = []

        self._actualPos = []  

        self._time = [0]
        self._pos.append([normalize(self._currentPos[i], self._posLimits[i]) for i in range(len(self._currentPos))])
        self._vel.append([normalize(self._currentVel[i], self._velLimits[i]) for i in range(len(self._currentVel))])
        self._acc.append([normalize(self._currentAcc[i], self._accLimits[i]) for i in range(len(self._currentAcc))])
        self._jerk.append(self._zero_vector.copy())  

        self._subTime = [0]
        self._subPos.append(self._pos[0].copy())
        self._subVel.append(self._vel[0].copy())
        self._subAcc.append(self._acc[0].copy())
        self._subJerk.append(self._jerk[0].copy())



    def displayPlot(self, maxTime=None):

        if not self._plotForPaper:
            numSubplots = 4
            if not self._plotForVideo:
                fig, ax = plt.subplots(numSubplots, 1, sharex=True)
                plt.subplots_adjust(left=0.05, bottom=0.04, right=0.95, top=0.98, wspace=0.15, hspace=0.15)
            else:
                pixel_width = int(0.4 * 1080 * 16 / 9)
                pixel_height = int(0.7 * 1080)
                dpi = 100  
                fig, ax = plt.subplots(numSubplots, 1, sharex=True, dpi=dpi)
                plt.subplots_adjust(left=0.1, bottom=0.08, right=0.98, top=0.98, wspace=0.15, hspace=0.23)
                fig.set_size_inches((pixel_width / dpi, pixel_height / dpi), forward=True)
                ax[-1].set_xlabel('Time [s]')

            axPos = 0
            axOffset = 1
            axVel = 0 + axOffset
            axAcc = 1 + axOffset
            axJerk = 2 + axOffset
            

        else:
            matplotlib.rc('lines', linewidth=1.2)
            matplotlib.rc('lines', markersize=3)

            numSubplots = 1
            fig, ax = plt.subplots(numSubplots, 1, sharex=True)
            plt.subplots_adjust(left=0.15, bottom=0.18, right=0.95, top=0.905, wspace=0.15, hspace=0.25)
            if numSubplots == 1:
                ax = [ax]

            axAcc = 0
            axPos = None
            axVel = None
            axJerk = None

        
        for i in range(len(ax)):
            ax[i].grid(True)
            if self._plotForVideo == 0:
                ax[i].set_xlabel('Time [s]')
            


        if axPos is not None:
            ax[axPos].set_ylabel('Position')

        if axVel is not None:
            ax[axVel].set_ylabel('Velocity')

        if axJerk is not None:
            ax[axJerk].set_ylabel('Jerk')

        if axAcc is not None:
            ax[axAcc].set_ylabel('Acceleration')


        jointPos = np.swapaxes(self._pos, 0, 1)  
        jointVel = np.swapaxes(self._vel, 0, 1)
        jointAcc = np.swapaxes(self._acc, 0, 1)
        jointJerk = np.swapaxes(self._jerk, 0, 1)

        if self._plotAccLimits:
            
            jointAccLimits = np.swapaxes(self._currentAccLimits, 0, 1)
            jointAccLimits = np.swapaxes(jointAccLimits, 1, 2)

        jointSubPos = np.swapaxes(self._subPos, 0, 1)  
        jointSubVel = np.swapaxes(self._subVel, 0, 1)
        jointSubAcc = np.swapaxes(self._subAcc, 0, 1)
        jointSubJerk = np.swapaxes(self._subJerk, 0, 1)

        if self._actualPos:
            
            actualTime = np.arange(0, len(self._actualPos)) * self._controlTimeStep
            jointActualPos = np.swapaxes(self._actualPos, 0, 1)


        linestyle = '-'

        if self._plotTimeLimits is not None:
            
            self._time = np.asarray(self._time) - self._plotTimeLimits[0]
            self._subTime = np.asarray(self._subTime) - self._plotTimeLimits[0]
            actualTime = actualTime - self._plotTimeLimits[0]

        if maxTime is None or maxTime >= self._time[-1]:
            timeMaxIndex = len(self._time)
        else:
            timeMaxIndex = np.argmin(np.asarray(self._time) <= maxTime)
        if maxTime is None or maxTime >= self._subTime[-1]:
            subTimeMaxIndex = len(self._subTime)
        else:
            subTimeMaxIndex = np.argmin(np.asarray(self._subTime) <= maxTime)

        if self._actualPos:
            if maxTime is None or maxTime >= actualTime[-1]:
                actualTimeMaxIndex = len(actualTime)
            else:
                actualTimeMaxIndex = np.argmin(np.asarray(actualTime) <= maxTime)


        
        for j in range(self._numJoints):
            print('Joint ' + str(j + 1) + ' (min/max)' +
                  ' Jerk: ' + str(np.min(jointSubJerk[j])) + ' / ' + str(np.max(jointSubJerk[j])) +
                  '; Acc: ' + str(np.min(jointSubAcc[j])) + ' / ' + str(np.max(jointSubAcc[j])) +
                  '; Vel: ' + str(np.min(jointSubVel[j])) + ' / ' + str(np.max(jointSubVel[j])) +
                  '; Pos: ' + str(np.min(jointSubPos[j])) + ' / ' + str(np.max(jointSubPos[j])))


        for j in range(self._numJoints):
            color = 'C' + str(j)  
            colorLimits = 'C' + str(j)
            if self._plotForVideo >= 2:
                colorLimits = '#01B702'  
                color = 'C0'
            marker = '.'
            if self._plotForVideo == 3:
                marker = 'None'

            if self._plotJoint[j]:
                label = 'Joint ' + str(j + 1)
                if axPos is not None:
                    ax[axPos].plot(self._time[:timeMaxIndex], jointPos[j][:timeMaxIndex], color=color, marker=marker, linestyle='None', label='_nolegend_')
                    ax[axPos].plot(self._subTime[:subTimeMaxIndex], jointSubPos[j][:subTimeMaxIndex], color=color, linestyle=linestyle, label=label)
                    if self._actualPos and self._plotActualValues:
                        actualPosPlot = normalize(jointActualPos[j], self._posLimits[j])
                        ax[axPos].plot(actualTime[:actualTimeMaxIndex], actualPosPlot[:actualTimeMaxIndex], color=color, linestyle=linestyle, label='_nolegend_')

                if axVel is not None:
                    ax[axVel].plot(self._time[:timeMaxIndex], jointVel[j][:timeMaxIndex], color=color, marker=marker, linestyle='None', label='_nolegend_')
                    ax[axVel].plot(self._subTime[:subTimeMaxIndex], jointSubVel[j][:subTimeMaxIndex], color=color, linestyle=linestyle, label=label)
                if self._actualPos and self._plotActualValues:
                    actualVel = np.diff(jointActualPos[j]) / self._controlTimeStep
                    actualVelPlot = normalize(actualVel, self._velLimits[j])
                    if axVel is not None:
                        ax[axVel].plot(actualTime[:max(actualTimeMaxIndex, 1)], [0] + list(actualVelPlot[:max(actualTimeMaxIndex-1, 0)]), color=color, linestyle=linestyle, label='_nolegend_')

                if self._plotForPaper == 1:
                    color = 'C3'  
                if axAcc is not None:
                    ax[axAcc].plot(self._time[:timeMaxIndex], jointAcc[j][:timeMaxIndex], color=color, marker=marker, linestyle='None', label='_nolegend_')
                    ax[axAcc].plot(self._subTime[:subTimeMaxIndex], jointSubAcc[j][:subTimeMaxIndex], color=color, linestyle=linestyle, label=label)

                if self._actualPos and self._plotActualValues:
                    actualAcc = np.diff(actualVel) / self._controlTimeStep
                    actualAccPlot = normalize(actualAcc, self._accLimits[j])
                    if axAcc is not None:
                        if self._plotForPaper == 1:
                            color = '#007800'  
                        ax[axAcc].plot(actualTime[:max(actualTimeMaxIndex-1, 1)], [0] + list(actualAccPlot[:max(actualTimeMaxIndex-2, 0)]), color=color, linestyle='--', label='_nolegend_')

                if axAcc is not None:
                    if self._plotAccLimits:
                        for i in range(2):
                            ax[axAcc].plot(self._time[:timeMaxIndex], jointAccLimits[j][i][:timeMaxIndex], color=colorLimits, linestyle='--', label='_nolegend_')

                if axJerk is not None:
                    ax[axJerk].plot(self._time[:timeMaxIndex], jointJerk[j][:timeMaxIndex], color=color, marker=marker, linestyle='None', label='_nolegend_')
                    ax[axJerk].plot(self._subTime[:subTimeMaxIndex], jointSubJerk[j][:subTimeMaxIndex], color=color, linestyle=linestyle, label=label)
                if self._actualPos and self._plotActualValues:
                    actualJerk = np.diff(actualAcc) / self._controlTimeStep
                    actualJerkPlot = normalize(actualJerk, self._jerkLimits[j])
                    if axJerk is not None:
                        ax[axJerk].plot(actualTime[:max(actualTimeMaxIndex-2, 1)], [0] + list(actualJerkPlot[:max(actualTimeMaxIndex-3, 0)]), color=color, linestyle=linestyle, label='_nolegend_')

        for i in range(len(ax)):
            if self._plotForPaper == 0 and self._plotForVideo == 0:
                ax[i].legend(loc='lower right')
            if self._plotTimeLimits is None:
                ax[i].set_xlim([0, self._time[-1]])  
            else:
                ax[i].set_xlim([0, self._plotTimeLimits[1] - self._plotTimeLimits[0]])

        if axAcc is not None:
            ax[axAcc].set_ylim([-1.05, 1.05])
        if axJerk is not None:
            ax[axJerk].set_ylim([-1.05, 1.05])

        fig.align_ylabels(ax)

        if self._plotForPaper:
            
            fig.align_ylabels(ax)
            if self._plotForPaper == 1:
                ax[axAcc].set_aspect(0.07)
                ax[axAcc].tick_params(pad=8)
                ax[axAcc].xaxis.labelpad = 15

            
            rp = rospkg.RosPack()
            evaluationDir = rp.get_path("iiwa_trajectory_evaluation")
            subFolder = 'plotForPaper_' + str(self._plotForPaper)
            plotString = 'timestep_' + '%.3f' % self._timeStep
            plotString += '_trajectory_' + str(self._trajectoryCounter)

            fig.savefig(os.path.join(evaluationDir, 'trajectory_plotter', subFolder, plotString + '.png'), dpi=250)
            fig.savefig(os.path.join(evaluationDir, 'trajectory_plotter', subFolder, plotString + '.eps'), format='eps', dpi=250)
            fig.savefig(os.path.join(evaluationDir, 'trajectory_plotter', subFolder, plotString + '.pdf'), format='pdf', dpi=250)


        if self._plotForVideo:
            ax[axPos].set_ylim([-1.1, 1.1])
            ax[axVel].set_ylim([-1.1, 1.1])
            ax[axAcc].set_ylim([-1.1, 1.1])
            ax[axJerk].set_ylim([-1.1, 1.1])
            fig.canvas.draw()
            rgb_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            rgb_array = rgb_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            matplotlib.pyplot.close(fig)
            return rgb_array

        
        
        fig.set_size_inches((24.1, 13.5), forward=False)
        print("Trajectory plotted. Close plot to continue")
        plt.show()



    def addDataPoint(self, normalized_acc, normalized_acc_range=None):
            
            

            self._time.append(self._time[-1] + self._timeStep)
            lastAcc = self._currentAcc.copy()
            lastVel = self._currentVel.copy()
            lastPos = self._currentPos.copy()
            self._currentAcc = [denormalize(normalized_acc[k], self._accLimits[k]) for k in
                                range(len(normalized_acc))]  
            self._currentJerk = [(self._currentAcc[k] - lastAcc[k]) / self._timeStep for k in range(len(self._currentAcc))]
            self._currentVel = [lastVel[k] + 0.5 * self._timeStep * (lastAcc[k] + self._currentAcc[k]) for k in
                                range(len(self._currentVel))]
            self._currentPos = [self._currentPos[k] + lastVel[k] * self._timeStep
                                + (1 / 3 * lastAcc[k] + 1 / 6 * self._currentAcc[k]) * self._timeStep ** 2
                                for k in range(len(self._currentPos))]

            self._pos.append([normalize(self._currentPos[k], self._posLimits[k]) for k in range(len(self._currentPos))])
            
            self._vel.append([normalize(self._currentVel[k], self._velLimits[k]) for k in range(len(self._currentVel))])
            
            self._jerk.append([normalize(self._currentJerk[k], self._jerkLimits[k]) for k in range(len(self._currentJerk))])
            
            self._acc.append(normalized_acc.tolist())

            
            self._currentAccLimits.append(normalized_acc_range)

            

            for j in range(1, self._plotNumSubTimeSteps + 1):
                t = j / self._plotNumSubTimeSteps * self._timeStep
                self._subTime.append(self._timeStepCounter * self._timeStep + t)
                self._subJerk.append(self._jerk[-1])  
                subCurrentAcc = [lastAcc[k] + ((self._currentAcc[k] - lastAcc[k]) / self._timeStep) * t
                                 for k in range(len(self._currentAcc))]
                subCurrentVel = [lastVel[k] + lastAcc[k] * t +
                                 0.5 * ((self._currentAcc[k] - lastAcc[k]) / self._timeStep) * t ** 2
                                 for k in range(len(self._currentVel))]
                subCurrentPos = [lastPos[k] + lastVel[k] * t + 0.5 * lastAcc[k] * t ** 2 +
                                 1 / 6 * ((self._currentAcc[k] - lastAcc[k]) / self._timeStep) * t ** 3
                                 for k in range(len(self._currentPos))]

                
                self._subAcc.append([normalize(subCurrentAcc[k], self._accLimits[k]) for k in range(len(subCurrentAcc))])
                self._subVel.append([normalize(subCurrentVel[k], self._velLimits[k]) for k in range(len(subCurrentVel))])
                self._subPos.append([normalize(subCurrentPos[k], self._posLimits[k]) for k in range(len(subCurrentPos))])

            self._timeStepCounter = self._timeStepCounter + 1

    def addActualPosition(self, actualPosition):
        
        
        self._actualPos.append(actualPosition)
        self._plotActualValues = True




def normalize(value, valueRange):
    
    
    normalizedValue = -1 + 2 * (value - valueRange[0]) / (valueRange[1] - valueRange[0])
    return normalizedValue


def denormalize(normValue, valueRange):
    
    
    actualValue = valueRange[0] + 0.5 * (normValue + 1) * (valueRange[1] - valueRange[0])
    return actualValue


