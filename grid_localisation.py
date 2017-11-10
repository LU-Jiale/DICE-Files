HALL_UNIT_DISPLACEMENT = 0.166775906
GRID_UNIT_DISTANCE = 10.0
MAX_X = 32
MAX_Y = 42

import numpy as np

class gridLocalisation:
    def __init__(self, IO):
        # Load map
        self.gridmap = np.loadtxt('gridMap.txt', dtype=float, delimiter=' ')
        self.gridmap = 1 - self.gridmap
        self.gridmap = self.gridmap[::-1]
        self.gridmap = self.gridmap.T

        #self.edges = edgeDetect(gridmap)
        self.edgeY = [[0,10], [1,10], [2,10], [3,10], [4,10], [5,10], [6,10], [7,10], [8,10], [9,10], [10,10],
                       [11,0], [12,0], [13,0], [14,0], [15,0], [16,0], [17,0], [18,0], [19,0], [20,0], [21,0],
                       [22,0], [23,0], [24,0], [25,0], [26,0], [27,0], [28,0], [29,1], [30,2], [31,3],
                       [7,41], [8,41], [9,41], [10,41],
                       [11,41], [12,41], [13,41], [14,41], [15,41], [16,41], [17,41], [18,41], [19,41], [20,41],
                       [21,41], [22,41], [23,41], [24,41], [25,41], [26,41], [27,41], [28,41], [29,41], [30,41],
                       [31,41], [0,35], [1,36], [2,37], [3,38], [4,39], [5,40], [6,40]]

        self.edgeX = [[11, 0], [11, 1], [11, 2], [11, 3], [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [11, 9],
                       [0,10], [0,11], [0,12], [0,13], [0,14], [0,15], [0,16], [0,17], [0,18], [0,19], [0,20],
                       [0,21], [0,22], [0,23], [0,24], [0,25], [0,26], [0,27], [0,28], [0,29], [0,30], [0,31],
                       [0,32], [0,33], [0,34], [0, 35], [1,36], [2,37], [3,38], [4,39], [5, 40], [6, 41],
                       [22,10], [22,11], [22,12], [22,13], [22,14], [22,15],
                       [22,16], [22,17], [22,18], [22,19], [22,20], [22,21], [22,22], [22,23], [22,24], [22,25],
                       [22,26], [22,27], [22,28], [22,29], [22,30], [22,31], [28,0], [29,1],
                       [30,2], [31,3], [31,4], [31,5], [31,6], [31,7], [31,8], [31,9], [31,10], [31,11], [31,12],
                       [31,13], [31,14], [31,15], [31,16], [31,17], [31,18], [31,19], [31,20], [31,21], [31,22],
                       [31,23], [31,24], [31,25], [31,26], [31,27], [31,28], [31,29], [31,30], [31,31], [31,32],
                       [31,33], [31,34], [31,35], [31,36], [31,37], [31,38], [31,39], [31,40], [31,41]]
        self.simpleGaussian = np.array([[0.025,0.1,0.025],[0.1,0.5,0.1],[0.025,0.1,0.025]])
        self.position = np.array([13, 4])
        self.positionMap = np.zeros([32, 42], dtype=float)
        self.positionMap[self.position[0], self.position[1]] = 1.0
        self.newPosition = np.array([5, 5])
        self.newPositionMap = np.zeros([32, 42], dtype=float)
        self.sensorMap = np.zeros([32, 42], dtype=float)
        self.orientation = 90
        self.movementX = 0
        self.movementY = 0

    def setOrientation(self, degree):
        self.orientation -= degree
        self.orientation %= 360

    def conv2(self, img, mask):
        imgShape = np.shape(img)
        ih = imgShape[0] + 2
        iw = imgShape[1] + 2
        enlargeImg = np.zeros([ih, iw])
        enlargeImg[1:ih - 1, 1: iw - 1] = img
        filterShape = np.shape(mask)
        result = np.zeros(imgShape)

        for x in np.arange(imgShape[0]):
            for y in np.arange(imgShape[1]):
                temp = enlargeImg[x-1:x+2, y-1:y+2]
                result[x,y] = np.correlate(temp.flat, mask.flat)[0]
                # for u in np.arange(filterShape[0]):
                #     for v in np.arange(filterShape[1]):
                #        result[x,y] += enlargeImg[x+u-1, y+v-1] * mask[u,v];
        return result

    def sensorMapGenerate(self, gridNumber):
        imgShape = np.shape(self.gridmap)
        sensorMap = np.zeros([32, 42], dtype=float)
        for edge_x in self.edgeX:
            [x, y] = edge_x
            if x >= gridNumber:
                if [x - gridNumber, y] in self.edgeX:
                    sensorMap[x,y] = 1.0
            if x < imgShape[0] - gridNumber:
                if [x + gridNumber, y] in self.edgeX:
                    sensorMap[x,y] = 1.0

        for edge_y in self.edgeX:
            [x, y] = edge_y
            if y >= gridNumber:
               if [x, y - gridNumber] in self.edgeY:
                    sensorMap[x,y] = 1.0
            if y < imgShape[1] - gridNumber:
                if [x, y + gridNumber] in self.edgeY:
                    sensorMap[x,y] = 1.0

        #sensorMap = self.conv2(sensorMap, self.simpleGaussian)
        sensorMap = sensorMap * (1.0 - self.gridmap)
        return sensorMap

    def senseNewPostition(self, front_ir, back_ir, sonar):
        #senseMap = np.zeros([32, 42], dtype=float)

        gridNumberLF = int (front_ir / GRID_UNIT_DISTANCE)
        gridNumberLB = back_ir / GRID_UNIT_DISTANCE
        gridNumberFront = (sonar - GRID_UNIT_DISTANCE / 2) / GRID_UNIT_DISTANCE
        IR_mapF = self.sensorMapGenerate(2)
        #IR_mapB = self.sensorMapGenerate(gridNumberLB)
        # sonar_map = self.sensorMapGenerate(gridNumberFront)
        senseMap = IR_mapF
        return senseMap

    def positionAfterMovement(self, displacement):
        mapTemp = np.zeros([32, 42], dtype=float)
        gridNumber = int((displacement + GRID_UNIT_DISTANCE / 2) / GRID_UNIT_DISTANCE)
        if self.orientation == 0:
            self.newPosition = self.position + [gridNumber, 0]
        elif self.orientation == 45:
            self.newPosition = self.position + [gridNumberSqrt, gridNumberSqrt]
        elif self.orientation == 90:
            self.newPosition = self.position + [0, gridNumber]
        elif self.orientation == 135:
            self.newPosition = self.position + [-gridNumberSqrt, gridNumberSqrt]
        elif self.orientation == 180:
            self.newPosition = self.position + [-gridNumber, 0]
        elif self.orientation == 225:
            self.newPosition = self.position + [-gridNumberSqrt, -gridNumberSqrt]
        elif self.orientation == 270:
            self.newPosition = self.position + [0, -gridNumber]
        elif self.orientation == 315:
            self.newPosition = self.position + [gridNumberSqrt, -gridNumberSqrt]

        mapTemp[self.newPosition[0], self.newPosition[1]] = 1.0
        #mapTemp = self.conv2(mapTemp, self.simpleGaussian)

        return mapTemp

    def gridUpdate(self, front_ir, back_ir, sonar, displacementN):
        self.newPositionMap = self.positionAfterMovement(int(displacementN))
        self.sensorMap = self.senseNewPostition(int(front_ir), int(back_ir), int(sonar))
        outPositionMap = self.newPositionMap * self.sensorMap

        #outPositionMap = outPositionMap / outPositionMap.max(axis = 0)
        index = np.where(outPositionMap == outPositionMap.max(axis = 0))
        self.position = np.array([index[0][0], index[1][0]])
        print self.position


