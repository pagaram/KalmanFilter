import numpy as np
import pandas as pd
import KalmanFunc as KF
import matplotlib.pyplot as plt

#reading in data
meas = pd.read_csv('LidarRadarData.txt',  header=None, delim_whitespace = True, skiprows=1)

#manually put it first row to intialize state and time (will find a better way later)
prev = 1477010443000000/1000000.0 #time stamp
X = np.array([[0.312242], [0.5803398], [0], [0]]) #x, y, vx, vy

#store calculated position and ground truth position
calc_pos = np.zeros([meas.shape[0], 2])
groundTruth_pos = np.zeros([meas.shape[0], 2])

#initialize state transtion matrix and covariance matrix
A = np.identity(4) #state transition

P = np.identity(4) #covariance matrix
P[2, 2] = 1000
P[3, 3] = 1000

#measurements transform and vector
H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]) #measurement transform (extracting x, y)
Z = np.zeros([2, 1]) #measurment vector (x, y)

#process noise and measurement noise matrix
Qv = np.identity(2) #process noise (acceleration is unknown)
Qv[0, 0] = 5 #ax, ay noise assumed to be uncorrelated
Qv[1, 1] = 5

G = np.zeros([4, 2]) #transform process noise

R = np.identity(2) #measurement noise
R[0, 0] = 0.0225 #x, y, noise assumned uncorrelated
R[1, 1] = 0.0225

#looping through data and applying the Kalman filter
step = -1
for i in range(len(meas)):
    new_meas = meas.iloc[i, :].values
    if new_meas[0] == 'L': #only taking lidar readings
        step = step + 1
        #computing delta time
        curr = new_meas[3]/1000000.0
        delta = curr - prev
        delta_squared = delta * delta
        prev = curr

        #update state transition
        A[0, 2] = delta
        A[1, 3] = delta

        #compute process noise
        G[0, 0] = delta_squared/2
        G[2, 0] = delta
        G[1, 1] = delta_squared/2
        G[3, 1] = delta
        Q = KF.processNoise(G, Qv)

        #reading in the measured values
        Z[0, 0] = new_meas[1]
        Z[1, 0] = new_meas[2]

        #now doing the kalman filter
        X, P = KF.predict(X, A, P, Q)
        K = KF.KalmanGain(P, H, R)
        X, P = KF.update(X, P, Z, K, H)

        #storing data
        calc_pos[step, :] = np.array([X[0, 0], X[1, 0]])
        groundTruth_pos[step, :] = np.array([new_meas[4], new_meas[5]])


#now plotting calculated position versus ground truth
calc_pos = calc_pos[0: step+1, :] #only keeping lidar data
groundTruth_pos = groundTruth_pos[0:step + 1, :]
starting = np.array([groundTruth_pos[0, 0], groundTruth_pos[0, 1]])
ending = np.array([groundTruth_pos[step, 0], groundTruth_pos[step, 1]])

plt.figure()
plt.plot(calc_pos[:, 0], calc_pos[:, 1], 'r-', label='Calculated Position')
plt.plot(groundTruth_pos[:, 0],  groundTruth_pos[:, 1], 'b-', label='Ground Truth')
plt.plot(starting[0], starting[1], 'k*', label='Start Point')
plt.plot(ending[0], ending[1], 'g*', label='End Point')
plt.xlabel('Position-X')
plt.ylabel('Position-Y')
plt.title('Kalman Filter Lidar Tracking')
plt.grid()
plt.legend()
plt.show()



