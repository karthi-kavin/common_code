# @title GET SHAPES
import numpy as np
from scipy.interpolate import interp1d
from bootstrap import bootstrap
#sizes are pre-scaled- assumes value for snz rather than normaling velocities after averaging.
def my_resize(vels,times,length):
      
      tmp = np.arange(1,length-1)/(length-1)
      outtimes = np.array([0] + list(tmp) + [1])
      mytimes = (np.array(times)-np.min(times))/(np.max(times)-np.min(times))
      f = interp1d(mytimes,vels)
      outvels = f(outtimes)

      return outtimes, outvels

def rescale(velocity, time, snzs, length):
 
 rtime, vel = my_resize(velocity, time, length)
 scalingF = 1/snzs -1
 rvel = np.array(vel)*(np.power(len(velocity)-2, -1*scalingF))

 return rtime, rvel

def meanNoEmpty(array):
  
  array2 = []
  for ar in array:
    if ar[1] != 0:
      array2.append(ar)

  mean = np.mean(array2, axis = 0)

  return mean

"""
gets scaled velocity profiles based on snz and the average velocity profile of the avalanches (within a given duration range).

input:
vels,time,durs,sizes are outputs of getSlips code.
minD: minimum duration of average
maxD: maximum duration of average
snzs: exponent from size vs duration, relates to scaling factor of sizes.
ifE: if 1 include error bars, if 0 dont calculate error
ci: confidence interval for error bars

Output:
scaled: array, every index is an avalanche (with duration within range) with sizes scaled by T^(1-1/snz)
t: time array corresponding to scaled avalanches, normalized to end at length 1 with maxduration+2 elements.
avgVel: average velocity curve of all avalanches with durations within range.
error: error bars for avgVel

"""
def avgScaledShape(vels, time, durs, sizes, snz, minD, maxD, ci = 0.95, ifE = 1):
  
  scaled = []
  index = 0

  for i in range(len(vels)):

    if minD <= durs[i] <= maxD:
      t, s = rescale(vels[i], time[i], snz, maxD+2)
      scaled.append(s)
      index += 1
      
  avgVel = np.mean(scaled, axis = 0)

  if index == 0:
    return 0
  
  if ifE == 1:
    astd = []
    scaledNumpy = np.array(scaled)

    for i in range(maxD+2):
      cur = scaledNumpy[:,i]
      conf = bootstrap((cur,),np.mean, confidence_level = ci).confidence_interval
      lo = avgVel[i] - conf[0]
      hi = conf[1] - avgVel[i]
      astd.append(np.array([lo,hi]))
    error = np.array(astd).transpose()

  else:
    error = []

  return scaled, t, avgVel, error


"""
first averages by each duration, then averages . Each duration bin is treated equally regardless of the number of avalanches within that bin in averaging.
While 1 and 2 week avalanches dominate the averages they do not provide interesting information about the shape (triangles and trapezoids respectfully), so this is useful in seeing shape contributions of the larger less frequent avalanches.

Inputs:
slipsData: output from getslips code
minD: minimum duration of average
maxD: maximum duration of average
snzs: exponent from size vs duration, relates to scaling factor of sizes.

Outputs:
time[0]: time array corresponding to first bin - all arrays in time array are the same since is always used maxD+2 in my_resize function
avgs: average shape of each avalanche - useful to plot all of these against the overall average
ShapeV: mean of avgs of each duration (each duration bin is weighted equally)
time: time array corresponding to bins, all are the same
hold: array of number of avalanches in bin


"""
def shapesAVGS(data, minD, maxD, snzs):

    scaleF = 1/snzs-1
    hold = [0]*(maxD-minD+1)
    sorted = [0]*(maxD-minD+1)
    avgs = [0]*(maxD-minD+1)
    time = [0]*(maxD-minD+1)

    for i in range(len(sorted)):
      
      hold[i] = 0
      time[i] = [0]*(i+2+minD)
      time[i][-1] = 1
      avgs[i] = np.array([0.0]*(i+2+minD))

      for k in range(len(time[i])-2):
        time[i][k+1] = (0.5+k)*(1/(i+minD))

      for j in range(len(data[3])):
        if data[3][j] == i+minD:
          hold[i] = hold[i] + 1
          avgs[i] += (np.array(data[0][j]))
          
      if hold[i] > 0:
        avgs[i] = avgs[i]/hold[i]*(np.power(i+minD, -1*scaleF))
      time[i], avgs[i] = my_resize(avgs[i], time[i], maxD+2)
    ShapeV = meanNoEmpty(avgs) #doesnt include empty arrays (all zeros) in average, which correspond to durations that have no avalanches
    return time[0], ShapeV, avgs, time, hold

