import numpy as np
import os
import inspect
import matplotlib
import math
import matplotlib.pyplot as plt
import jointLimitEquations
from datetime import datetime
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from functools import partial
from multiprocessing import Pool
import timeit
from importlib import import_module

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

currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

class PosVelJerkLimitation():
    def __init__(self,
                 timeStep,
                 posLimits,
                 velLimits,
                 accLimits,
                 jerkLimits,
                 accelerationAfterMaxVelLimitFactor=0.0001,
                 setVelocityAfterMaxPosToZero=True,
                 limitVelocity=True,
                 limitPosition=True,
                 numWorkers=1,
                 *vargs,
                 **kwargs):

        self._timeStep = timeStep
        self._posLimits = posLimits
        self._numJoints = len(self._posLimits)
        self._velLimits = velLimits
        self._accLimits = accLimits
        self._jerkLimits = jerkLimits

        self._accelerationAfterMaxVelLimitFactor = accelerationAfterMaxVelLimitFactor
        self._setVelocityAfterMaxPosToZero = setVelocityAfterMaxPosToZero
        self._limitVelocity = limitVelocity
        self._limitPosition = limitPosition

        self._workerPool = None  
        jointLimitEquations.jointLimitColdStart()

        if numWorkers >= 2 and self._numJoints >= 2:
            self._workerPool = Pool(min(numWorkers, self._numJoints))
        else:
            self._workerPool = None


    @property
    def setVelocityAfterMaxPosToZero(self):
        return self._setVelocityAfterMaxPosToZero

    @setVelocityAfterMaxPosToZero.setter
    def setVelocityAfterMaxPosToZero(self, setVelocityAfterMaxPosToZero):
        self._setVelocityAfterMaxPosToZero = setVelocityAfterMaxPosToZero


    def calculateValidAccelerationRange(self, currentPos, currentVel, currentAcc, timeStepCounter=0):

        if self._workerPool:
            poolResult = self._workerPool.starmap(self._calculateValidAccelerationRangePerJoint,
                                            [(i, self._timeStep, currentPos[i], currentVel[i], currentAcc[i],
                                              self._posLimits[i], self._velLimits[i], self._accLimits[i],
                                              self._jerkLimits[i],
                                              self._accelerationAfterMaxVelLimitFactor,
                                              self._setVelocityAfterMaxPosToZero,
                                              False,
                                              self._limitVelocity, self._limitPosition, timeStepCounter)
                                             for i in range(self._numJoints)])

            poolResult = np.swapaxes(poolResult, 0, 1)
            normAccRange = poolResult[0]
            limitViolation = poolResult[1]

        else:
            normAccRange = []  
            limitViolation = []

            for i in range(self._numJoints):
                
                normAccRangeJoint, limitViolationJoint = self._calculateValidAccelerationRangePerJoint(i, self._timeStep, currentPos[i], currentVel[i],
                                                                  currentAcc[i], self._posLimits[i], self._velLimits[i],
                                                                  self._accLimits[i], self._jerkLimits[i],
                                                                  self._accelerationAfterMaxVelLimitFactor,
                                                                  self._setVelocityAfterMaxPosToZero,
                                                                  False,
                                                                  self._limitVelocity, self._limitPosition,
                                                                  timeStepCounter)

                normAccRange.append(normAccRangeJoint)
                limitViolation.append(limitViolationJoint)
                

        return normAccRange, limitViolation


    
    def _calculateValidAccelerationRangePerJoint(self, jointIndex, tS, currentPos, currentVel, currentAcc, posLimits,
                                                 velLimits, accLimits, jerkLimits, accelerationAfterMaxVelLimitFactor,
                                                 setVelocityAfterMaxPosToZero=False,
                                                 clipAccelerationAfterMaxPosToZero=False,
                                                 limitVelocity=True, limitPosition=True,
                                                 timeStepCounter=0):

        
        accRangeJerk = [currentAcc + jerkLimits[0] * tS,
                        currentAcc + jerkLimits[1] * tS]  
        accRangeAcc = [accLimits[0], accLimits[1]]  
        
        
        
        accRangeStaticVel = [(v1 - currentVel) / (0.5 * tS) - currentAcc for v1 in velLimits]
        

        
        accRangeDynamicVel = [accLimits[0], accLimits[1]]

        if limitVelocity:
            
            if (currentAcc < 0 and (
                    currentVel < velLimits[0] + 0.5 * (currentAcc ** 2 * tS) / (accLimits[1] - currentAcc))):
                
                
                accRangeDynamicVel = [accLimits[1], accLimits[1]]
            else:
                if (currentAcc > 0 and (
                        currentVel > velLimits[1] - 0.5 * (currentAcc ** 2 * tS) / (
                        currentAcc - accLimits[0]))):
                    
                    
                    accRangeDynamicVel = [accLimits[0], accLimits[0]]
                else:
                    for j in range(2):
                        
                        nj = (
                                        j + 1) % 2  
                        

                        if (j == 0 and (currentVel + 0.5 * currentAcc * tS) <= velLimits[0]) \
                                or (j == 1 and (currentVel + 0.5 * currentAcc * tS) >= velLimits[1]):
                            

                            accRangeDynamicVel[j] = currentAcc * (
                                    1 - ((0.5 * currentAcc * tS) / (velLimits[j] - currentVel)))

                        else:
                            

                            a = - jerkLimits[nj] / 2
                            b = tS * jerkLimits[nj] / 2
                            c = currentVel - velLimits[j] + currentAcc * tS / 2

                            if a == 0:
                                raise ValueError('Jerk limits are not allowed to be zero')
                            if b ** 2 - 4 * a * c >= 0:
                                if j == 0:
                                    t_a0_1 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (
                                            2 * a)  
                                else:
                                    t_a0_1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (
                                            2 * a)  

                                
                                a1Limit = - jerkLimits[nj] * (t_a0_1 - tS)

                                if np.ceil(t_a0_1 / tS) > t_a0_1 / tS:
                                    
                                    
                                    n = np.ceil(t_a0_1 / tS) - 1
                                    
                                    a_n_plus_1 = a1Limit + jerkLimits[nj] * tS * n
                                    
                                    if (j == 0 and a_n_plus_1 > accLimits[1] * accelerationAfterMaxVelLimitFactor) \
                                            or (j == 1 and
                                                a_n_plus_1 < accLimits[0] * accelerationAfterMaxVelLimitFactor):

                                        a_0 = currentAcc
                                        a_n_plus_1_star = accLimits[nj] * accelerationAfterMaxVelLimitFactor
                                        t_n = n * tS
                                        j_min = jerkLimits[nj]
                                        
                                        v_0 = currentVel
                                        v_max = velLimits[j]

                                        a1Limit = jointLimitEquations.velocityReducedAcceleration(j, j_min, a_0,
                                                                                                    a_n_plus_1_star,
                                                                                                    v_0, v_max, tS, t_n)

                                accRangeDynamicVel[j] = a1Limit

        logTimestep = []
        logJointIndex = []
        logJ = []

        accRangeDynamicPos = [-10 ** 6, 10 ** 6]

        if limitPosition:

            for j in range(2):
                
                nj = (j + 1) % 2  

                a_min = accLimits[nj]

                j_min = jerkLimits[nj]
                j_max = jerkLimits[j]

                p_max = posLimits[j]
                v_max = velLimits[j]

                p_0 = currentPos
                v_0 = currentVel
                a_0 = currentAcc

                a_1_min_first = 0  
                t_v0_min_first = 0
                t_a_min_min_jerk = 0
                t_n_a_min_min_jerk = 0
                a_1_upper_bound = 0  
                t_v0_upper_bound = 0
                t_a_min_upper_bound = 0
                t_n_a_min = 0
                a_1_all_phases = 0
                t_v0_all_phases = None
                a_1_reduced_jerk = None
                t_v0_reduced_jerk = 0

                plotDynStagePosMode = 0  

                t_v0_bounded_vel_min_jerk_phase = None  
                
                t_n_u_phase = None
                if setVelocityAfterMaxPosToZero:
                    t_star_all_phases = None
                    t_n_u_all_phases = None
                    a_1_bounded_vel_continuous_all_phases = None
                    t_u_bounded_vel_continuous_all_phases = None
                    a_1_bounded_vel_discrete_all_phases = None
                    j_n_u_plus_1_all_phases = None
                    t_v0_bounded_vel_min_jerk_phase = None
                    t_star_min_jerk_phase = None
                    t_n_u_min_jerk_phase = None
                    a_1_bounded_vel_continuous_min_jerk = None
                    t_u_bounded_vel_continuous_min_jerk = None
                    a_1_bounded_vel_discrete_min_jerk = None
                    j_n_u_plus_1_min_jerk_phase = None
                    t_v0_bounded_acc = None  

                
                a_1_min_jerk, t_v0_min_jerk = jointLimitEquations.positionBorderCaseMinJerkPhase(j_min, a_0, v_0, p_0,
                                                                                                    p_max, tS)

                if t_v0_min_jerk < tS + 1e-8 or math.isnan(t_v0_min_jerk):
                    
                    a_1_min_first, t_v0_min_first = jointLimitEquations.positionBorderCaseFirstPhase(j, a_0, v_0, p_0,
                                                                                                        p_max,
                                                                                                        tS)
                    if j == plotDynStagePosMode:
                        dynPosStage = -1

                    if 0 < t_v0_min_first <= tS + 1e-3:
                        accRangeDynamicPos[j] = a_1_min_first
                        if j == plotDynStagePosMode:
                            dynPosStage = 0
                    else:

                        if p_0 == p_max and a_0 == 0 and v_0 == 0:
                            accRangeDynamicPos[j] = 0


                else:

                    t_n_a_min = tS * (1 + np.floor((a_min - a_1_min_jerk) / (j_min * tS)))

                    if t_n_a_min >= t_v0_min_jerk:
                        if t_v0_min_jerk >= tS:
                            
                            accRangeDynamicPos[j] = a_1_min_jerk
                            if j == plotDynStagePosMode:
                                dynPosStage = 1
                            

                            if setVelocityAfterMaxPosToZero:
                                t_v0_bounded_vel_min_jerk_phase = t_v0_min_jerk

                    else:

                        t_a_min = tS * (1 + ((a_min - a_1_min_jerk) / (j_min * tS)))

                        t_a_min_min_jerk = t_a_min
                        t_n_a_min_min_jerk = t_n_a_min

                        if t_v0_min_jerk > t_a_min:


                            a_1_upper_bound, t_v0_upper_bound = jointLimitEquations.positionBorderCaseUpperBound(j,
                                                                                                                    j_min,
                                                                                                                    a_0,
                                                                                                                    a_min,
                                                                                                                    v_0,
                                                                                                                    p_0,
                                                                                                                    p_max,
                                                                                                                    tS)

                            if math.isnan(a_1_upper_bound):
                                accRangeDynamicPos[j] = a_0 + j_min
                                continue
                                                                                    

                            t_a_min_upper_bound = tS * (1 + ((a_min - a_1_upper_bound) / (j_min * tS)))
                            if t_a_min_upper_bound < tS:
                                
                                if t_a_min_upper_bound / tS > 0.999:
                                    t_a_min_upper_bound = tS
                                    
                            t_n_a_min = tS * np.floor(t_a_min_upper_bound / tS)


                        a_1_all_phases, t_v0_all_phases = jointLimitEquations.positionBorderCaseAllPhases(j, j_min, a_0,
                                                                                                            a_min, v_0,
                                                                                                            p_0,
                                                                                                            p_max, tS,
                                                                                                            t_n_a_min)
                        if t_v0_all_phases >= t_n_a_min + tS:
                            accRangeDynamicPos[j] = a_1_all_phases
                            if j == plotDynStagePosMode:
                                dynPosStage = 2

                            if setVelocityAfterMaxPosToZero:
                                
                                t_star_all_phases = tS * np.ceil(
                                    t_v0_all_phases / tS)  
                                a_1_bounded_vel_continuous_all_phases, t_u_bounded_vel_continuous_all_phases =  \
                                    jointLimitEquations.positionBoundedVelocityContinuousAllPhases(j_min, j_max, a_0,
                                                                                                    a_min, v_0, p_0,
                                                                                                    p_max, tS,
                                                                                                    t_star_all_phases,
                                                                                                    t_n_a_min)
                                
                                if t_u_bounded_vel_continuous_all_phases >= t_star_all_phases:
                                    print("t_u too big for joint " + str(jointIndex))  
                                    

                                if t_u_bounded_vel_continuous_all_phases >= t_n_a_min + tS:
                                    
                                    t_n_u_all_phases = tS * np.floor(t_u_bounded_vel_continuous_all_phases / tS)
                                    a_1_bounded_vel_discrete_all_phases, j_n_u_plus_1_all_phases = \
                                        jointLimitEquations.positionBoundedVelocityDiscreteAllPhases(j_min, j_max, a_0,
                                                                                                        a_min, v_0, p_0,
                                                                                                        p_max, tS,
                                                                                                        t_star_all_phases,
                                                                                                        t_n_a_min,
                                                                                                        t_n_u_all_phases)

                                    a_n_a_min = a_1_bounded_vel_discrete_all_phases + (t_n_a_min - tS) * j_min
                                    if (j == 0 and a_n_a_min > a_min + 1e-3) or (j == 1 and a_n_a_min < a_min - 1e-3):
                                        

                                        if round(t_n_a_min / tS) > 1:
                                            
                                            a_1_bounded_vel_discrete_all_phases, j_n_u_plus_1_all_phases = \
                                                jointLimitEquations.positionBoundedVelocityDiscreteAllPhases(j_min,
                                                                                                                j_max,
                                                                                                                a_0,
                                                                                                                a_min, v_0,
                                                                                                                p_0,
                                                                                                                p_max, tS,
                                                                                                                t_star_all_phases,
                                                                                                                t_n_a_min - tS,
                                                                                                                t_n_u_all_phases)

                                            a_n_a_min = a_1_bounded_vel_discrete_all_phases + (
                                                        t_n_a_min - 2 * tS) * j_min

                                    if (j == 0 and a_n_a_min > a_min + 1e-3) or \
                                            (j == 1 and a_n_a_min < a_min - 1e-3):

                                        pass
                                    else:

                                        if (j == 0 and j_max <= j_n_u_plus_1_all_phases <= 0) or  (j == 1 and 0 <= j_n_u_plus_1_all_phases <= j_max):

                                            if (j == 0 and a_1_bounded_vel_discrete_all_phases > a_1_all_phases) or (j == 1 and a_1_bounded_vel_discrete_all_phases < a_1_all_phases):

                                                accRangeDynamicPos[j] = a_1_bounded_vel_discrete_all_phases
                                            else:  

                                                t_v0_bounded_acc = t_v0_all_phases

                                        else:
                                            pass
                                else:
                                    t_v0_bounded_vel_min_jerk_phase = t_v0_all_phases

                        else:

                            if setVelocityAfterMaxPosToZero:

                                t_v0_bounded_vel_min_jerk_phase = t_v0_all_phases

                            a_1_reduced_jerk, t_v0_reduced_jerk = jointLimitEquations.positionBorderCaseReducedJerkPhase(j_min, a_0, a_min, v_0, p_0, p_max, tS, t_n_a_min)

                            accRangeDynamicPos[j] = a_1_reduced_jerk

                            if j == plotDynStagePosMode:
                                dynPosStage = 3

                    
                    if setVelocityAfterMaxPosToZero and t_v0_bounded_vel_min_jerk_phase:

                        t_star_min_jerk_phase = tS * np.ceil(
                            t_v0_bounded_vel_min_jerk_phase / tS)  

                        if t_star_min_jerk_phase >= 3 * tS:
                            a_1_bounded_vel_continuous_min_jerk, t_u_bounded_vel_continuous_min_jerk = \
                                jointLimitEquations.positionBoundedVelocityContinuousMinJerkPhase(j_min, j_max, a_0,
                                                                                                    v_0,
                                                                                                    p_0, p_max, tS,
                                                                                                    t_star_min_jerk_phase)

                            if not math.isnan(t_u_bounded_vel_continuous_min_jerk) and \
                                    (t_u_bounded_vel_continuous_min_jerk / tS) > 0.99:
                                
                                t_n_u_min_jerk_phase = max(tS * np.floor(t_u_bounded_vel_continuous_min_jerk / tS), tS)
                            else:
                                t_n_u_min_jerk_phase = float('nan')

                        else:
                            t_n_u_min_jerk_phase = tS  

                        if not math.isnan(t_n_u_min_jerk_phase):
                            a_1_bounded_vel_discrete_min_jerk, j_n_u_plus_1_min_jerk_phase = \
                                jointLimitEquations.positionBoundedVelocityDiscreteMinJerkPhase(j_min, j_max, a_0, v_0,
                                                                                                p_0, p_max, tS,
                                                                                                t_star_min_jerk_phase,
                                                                                                t_n_u_min_jerk_phase)
                            if (j == 0 and j_max - 1e-6 <= j_n_u_plus_1_min_jerk_phase <= j_min + 1e-6) or \
                                    (j == 1 and j_min - 1e-6 <= j_n_u_plus_1_min_jerk_phase <= j_max + 1e-6):

                                if t_star_all_phases is not None:
                                    if (j == 0 and a_1_bounded_vel_discrete_min_jerk > a_1_all_phases) or \
                                            (j == 1 and a_1_bounded_vel_discrete_min_jerk < a_1_all_phases):

                                        accRangeDynamicPos[j] = a_1_bounded_vel_discrete_min_jerk
                                    else:
                                        
                                        t_v0_bounded_acc = t_v0_bounded_vel_min_jerk_phase
                                else:
                                    if a_1_reduced_jerk is not None:
                                        if (j == 0 and a_1_bounded_vel_discrete_min_jerk > a_1_reduced_jerk) or \
                                                (j == 1 and a_1_bounded_vel_discrete_min_jerk < a_1_reduced_jerk):
                                            accRangeDynamicPos[j] = a_1_bounded_vel_discrete_min_jerk
                                        else:
                                            
                                            t_v0_bounded_acc = t_v0_bounded_vel_min_jerk_phase
                                    else:

                                        if (j == 0 and a_1_bounded_vel_discrete_min_jerk > a_1_min_jerk) or \
                                                (j == 1 and a_1_bounded_vel_discrete_min_jerk < a_1_min_jerk):
                                            accRangeDynamicPos[j] = a_1_bounded_vel_discrete_min_jerk
                                        else:
                                            
                                            t_v0_bounded_acc = t_v0_bounded_vel_min_jerk_phase

                            else:
                                t_v0_bounded_acc = t_v0_bounded_vel_min_jerk_phase
                                
                        else:
                            t_v0_bounded_acc = t_v0_bounded_vel_min_jerk_phase

                if math.isnan(accRangeDynamicPos[j]):
                    accRangeDynamicPos[j] = accLimits[j]

        accRangeList = []  

        accRangeList.append(accRangeJerk)  
        accRangeList.append(accRangeAcc)

        if limitVelocity:
            accRangeList.append(accRangeDynamicVel)

        if limitPosition:
            accRangeList.append(accRangeDynamicPos)

        limitViolationCode = 0  

        for j in range(len(accRangeList)):
            if j <= limitViolationCode:
                accRangeSwap = np.swapaxes(accRangeList, 0, 1)
                accRangeTotal = [np.max(accRangeSwap[0][j:]), np.min(accRangeSwap[1][j:])]
                if math.isnan(accRangeTotal[0]) or math.isnan(accRangeTotal[1]) or \
                        (accRangeTotal[0] - accRangeTotal[1]) > 0.001:
                    limitViolationCode = limitViolationCode + 1
                else:
                    if (accRangeTotal[0] - accRangeTotal[1]) > 0:
                        
                        accRangeTotal[1] = accRangeTotal[0]

        normAccRangeJoint = [normalize(accLimit, accLimits) for accLimit in accRangeTotal]
        normAccRangeJoint = list(np.clip(normAccRangeJoint, -1, 1))

        return normAccRangeJoint, limitViolationCode


def normalize(value, valueRange):
    
    
    normalizedValue = -1 + 2 * (value - valueRange[0]) / (valueRange[1] - valueRange[0])
    return normalizedValue


def denormalize(normValue, valueRange):
    
    
    actualValue = valueRange[0] + 0.5 * (normValue + 1) * (valueRange[1] - valueRange[0])
    return actualValue


