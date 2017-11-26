class Car:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.collisionbox = [[], []]

    def GetVel(self):
        return self.velocity

    def GetPos(self):
        return self.position

    def updatePos(self):
        self.collisionbox[0] = [self.position-1, self.position +1]
        self.position += self.velocity
        if self.position >=1000:
            self.collisionbox[0] = [0,0]
            self.collisionbox[1] = [0, 0]
        else:
            self.collisionbox[1] = [self.position-1, self.position+1]
    def CollisionBox(self):
        return self.collisionbox


class PlayerCar(Car):
    def __init__(self, position, velocity):
        super().__init__(position, velocity)
        self.velchange = 0
    def updatePos(self):
        self.collisionbox[0] = [self.position - 1, self.position + 1]
        self.position += self.velocity
        self.collisionbox[1] = [self.position - 1, self.position + 1]

    def updateVeloc(self, inputvel):
        temp = self.velocity
        if (inputvel != 0):
            self.velocity = inputvel
            self.velchange = self.velocity - temp


