HALL_UNIT_DISPLACEMENT = 0.166775906
GRID_UNIT_DISTANCE = 10.0
MAX_X = 32
MAX_Y = 42

import numpy as np

class gridLocalisation:
    def __init__(self):
        # Load map
        self.gridmap = np.loadtxt('gridMap.txt', dtype=float, delimiter=' ')
        self.gridmap = 1 - self.gridmap
        self.gridmap = self.gridmap[::-1]
        self.gridmap = self.gridmap.T

        self.position = np.array([13, 10])
        self.positionMap = np.ones([32, 42], dtype=float) * 0.001
        self.positionMap[self.position[0], self.position[1]] = 1.0

        self.newPosition = np.array([5, 5])

        self.routeMap = np.zeros([32, 42], dtype=float)
        self.routeMap[self.position[0], self.position[1]] = 1.0

        self.sensorMap = np.zeros([32, 42], dtype=float)

        self.orientation = 180
        self.movementX = 0
        self.movementY = 0

    def setOrientation(self, degree):
        self.orientation -= degree
        self.orientation %= 360

    def positionAfterMovement(self, displacement):
        gridNumber = int((displacement + GRID_UNIT_DISTANCE / 2) / GRID_UNIT_DISTANCE)
        if self.orientation == 0:
            self.newPosition = self.position + [gridNumber, 0]
            self.positionMap = np.roll(self.positionMap.T, gridNumber).T
        elif self.orientation == 45:
            self.newPosition = self.position + [gridNumberSqrt, gridNumberSqrt]
        elif self.orientation == 90:
            self.newPosition = self.position + [0, gridNumber]
            self.positionMap = np.roll(self.positionMap, gridNumber)
        elif self.orientation == 135:
            self.newPosition = self.position + [-gridNumberSqrt, gridNumberSqrt]
        elif self.orientation == 180:
            self.newPosition = self.position + [-gridNumber, 0]
            self.positionMap = np.roll(self.positionMap.T, -gridNumber).T
        elif self.orientation == 225:
            self.newPosition = self.position + [-gridNumberSqrt, -gridNumberSqrt]
        elif self.orientation == 270:
            self.newPosition = self.position + [0, -gridNumber]
            self.positionMap = np.roll(self.positionMap, -gridNumber)
        elif self.orientation == 315:
            self.newPosition = self.position + [gridNumberSqrt, -gridNumberSqrt]

        # if self.newPosition[0] < 31:
        #     self.positionMap[self.newPosition[0] + 1, self.newPosition[1]] += 0.1
        # if self.newPosition[0] > 0:
        #     self.positionMap[self.newPosition[0] - 1, self.newPosition[1]] += 0.1
        # if self.newPosition[1] < 41:
        #     self.positionMap[self.newPosition[0], self.newPosition[1] + 1] += 0.1
        # if self.newPosition[1] > 0:
        #     self.positionMap[self.newPosition[0], self.newPosition[1] - 1] += 0.1

    def senseNewPostition(self, front_ir, back_ir, sonar):
        prob = 0.5
        senseMap = np.ones([32, 42], dtype=float) * 0.02 * prob
        gridNumberL = int ((front_ir + back_ir +  GRID_UNIT_DISTANCE) *0.5 / GRID_UNIT_DISTANCE)
        gridNumberFront = int((sonar + GRID_UNIT_DISTANCE / 2) / GRID_UNIT_DISTANCE)
        if gridNumberL <= 8:
            if self.orientation == 90:
                senseMap[11 + gridNumberL, 1:10] += prob
                senseMap[10 + gridNumberL, 1:10] += 0.1 * prob
                senseMap[12 + gridNumberL, 1:10] += 0.1 * prob
                senseMap[0 + gridNumberL, 11:35] += prob
                senseMap[1 + gridNumberL, 11:35] += 0.1 * prob
                senseMap[-1 + gridNumberL, 11:35] += 0.1 * prob
            elif self.orientation == 270:
                if gridNumberL <= 5:
                    senseMap[31 - gridNumberL, 4:41] += prob
                    senseMap[30 - gridNumberL, 4:41] += 0.1 * prob
                    senseMap[32 - gridNumberL, 4:41] += 0.1 * prob
                    senseMap[22 - gridNumberL, 10:32] += prob
                    senseMap[21 - gridNumberL, 10:32] += 0.1 * prob
                    senseMap[23 - gridNumberL, 10:32] += 0.1 * prob
                else:
                    senseMap[22 - gridNumberL, 9:24] += prob
                    senseMap[21 - gridNumberL, 9:24] += 0.1 * prob
                    senseMap[23 - gridNumberL, 9:24] += 0.1 * prob

            elif self.orientation == 0:
                senseMap[7 : 30, 41 - gridNumberL] += prob
                senseMap[7 : 30, 40 - gridNumberL] += 0.1 * prob
                senseMap[7 : 30, 42 - gridNumberL] += 0.1 * prob

            elif self.orientation == 180:
                senseMap[10:29, 0 + gridNumberL] += prob
                senseMap[10:29, 1 + gridNumberL] += 0.1 * prob
                senseMap[10:29, -1 + gridNumberL] += 0.1 * prob
                senseMap[0:11, 10 + gridNumberL] += prob
                senseMap[0:11, 11 + gridNumberL] += 0.1 * prob
                senseMap[0:11, 9 + gridNumberL] += 0.1 * prob

        if gridNumberFront < 18:
            if self.orientation == 90:
                senseMap[11:22, 24 - gridNumberFront] += prob
                senseMap[11:22, 23 - gridNumberFront] += 0.1 * prob
                senseMap[11:22, 25 - gridNumberFront] += 0.1 * prob
                senseMap[0:9, 36 - gridNumberFront] += prob
                senseMap[0:9, 35 - gridNumberFront] += 0.1 * prob
                senseMap[0:9, 37 - gridNumberFront] += 0.1 * prob
            elif self.orientation == 270:
                senseMap[16:22, 0 + gridNumberFront] += prob
                senseMap[16:22, 1 + gridNumberFront] += 0.1 * prob
                senseMap[16:22, -1 + gridNumberFront] += 0.1 * prob
                senseMap[26:31, 3 + gridNumberFront] += prob
                senseMap[26:31, 2 + gridNumberFront] += 0.1 * prob
                senseMap[26:31, 4 + gridNumberFront] += 0.1 * prob

            elif self.orientation == 0:
                senseMap[22 - gridNumberFront, 31:36] += prob
                senseMap[21 - gridNumberFront, 31:36] += 0.1 * prob
                senseMap[23 - gridNumberFront, 31:36] += 0.1 * prob
                senseMap[31 - gridNumberFront, 34:41] += prob
                senseMap[30 - gridNumberFront, 34:41] += 0.1 * prob
                senseMap[32 - gridNumberFront, 34:41] += 0.1 * prob

            elif self.orientation == 180:
                senseMap[11 + gridNumberFront, 0 : 10] += prob
                senseMap[10 + gridNumberFront, 0 : 10] += 0.1 * prob
                senseMap[12 + gridNumberFront, 0 : 10] += 0.1 * prob
                senseMap[0 + gridNumberFront, 10 : 23] += prob
                senseMap[1 + gridNumberFront, 10 : 23] += 0.1 * prob
                senseMap[-1 + gridNumberFront, 10 : 23] += 0.1 * prob
        return senseMap * (1 - self.gridmap)
        return senseMap * (1 - self.gridmap)

    def gridUpdate(self, front_ir, back_ir, sonar, displacementN):
        self.positionAfterMovement(int(displacementN))
        self.sensorMap = self.senseNewPostition(front_ir, back_ir, sonar)
        self.positionMap *= self.sensorMap
        self.positionMap = np.correlate(self.positionMap.flatten(),
                                        np.array([0.05, 0.2, 1, 0.2, 0.01]), 'same').reshape(32,42)
        self.positionMap = np.correlate(self.positionMap.T.flatten(),
                                        np.array([0.05, 0.2, 1, 0.2, 0.01]), 'same').reshape(42,32).T
        self.positionMap /= np.amax(self.positionMap)
        index = np.where(self.positionMap == np.amax(self.positionMap))
        self.position = np.array([index[0][0], index[1][0]])
        self.routeMap[self.position[0], self.position[1]] = 1.0
        print self.position

import matplotlib.pyplot as plt
from time import clock
t1 = gridLocalisation()
for i in np.arange(15):
    if i == 0:
        map1 = t1.senseNewPostition(20.0, 20.0, 200.0)
        t1.positionAfterMovement(0)
        plt.subplot(131)
        p1 = plt.imshow(map1)
        plt.subplot(132)
        p3 = plt.imshow(t1.positionMap)
        plt.subplot(133)
        p2 = plt.imshow(t1.routeMap)
        fig = plt.gcf()
        plt.clim()  # clamp the color limits
        plt.title("Robot")
    else:
        t1.gridUpdate(20,20,200 - 13*i, 10)
        p1.set_data(t1.sensorMap)
        p3.set_data(t1.positionMap)
        p2.set_data(t1.routeMap)
    plt.pause(1)
t1.orientation = 90
for i in np.arange(7):
    t1.gridUpdate(20, 20, 280 - 13 * i, 10)
    p1.set_data(t1.sensorMap)
    p3.set_data(t1.positionMap)
    p2.set_data(t1.routeMap)
    plt.pause(1)
for i in np.arange(7):
    t1.gridUpdate(100, 100, 280 - 13 * (i + 6), 10)
    p1.set_data(t1.sensorMap)
    p3.set_data(t1.positionMap)
    p2.set_data(t1.routeMap)
    plt.pause(1)
t1.orientation = 180
for i in np.arange(7):
    t1.gridUpdate(30, 30, 120 - 13 * i, 10)
    p1.set_data(t1.sensorMap)
    p3.set_data(t1.positionMap)
    p2.set_data(t1.routeMap)
    plt.pause(1)
t1.orientation = 90
for i in np.arange(20):
    t1.gridUpdate(30, 30, 300 - 13 * i, 10)
    p1.set_data(t1.sensorMap)
    p3.set_data(t1.positionMap)
    p2.set_data(t1.routeMap)
    plt.pause(1)
plt.pause(50)