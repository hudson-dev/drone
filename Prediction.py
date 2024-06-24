class Prediction:
    global prevArea, area, dist

    prevArea = -1;

    def __init__(self):
        pass
    
    def updateDist(xDist, x1, x2, y1, y2):
        area = Math.abs((x1-x2)*(y1-y2))
        if prevArea == -1:
            prevArea = area
        else:
            dist = xDist / (1-Math.sqrt(area/prevArea))
            prevArea = area

    def getDist(): 
        return dist