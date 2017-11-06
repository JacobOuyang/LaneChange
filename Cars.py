class Car:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity


    def GetVel(self):
        return self.velocity

    def GetPos(self):
        return self.position

    def updatePos(self):
        self.position += self.velocity
    def CollisionBox(self):
        return [self.position-1, self.position, self.position+1]


class PlayerCar(Car):
    #def __init__(self, position, velocity):
     #   super.__init__(position, velocity)

    def updateVeloc(self, inputvel):
        if (inputvel != 0):
            self.velocity = inputvel

