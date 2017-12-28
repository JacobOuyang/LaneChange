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
    def updateVeloc(self, inputvel):

        self.velocity = inputvel


class PlayerCar(Car):
    def __init__(self, position, velocity):
        super().__init__(position, velocity)
        self.velchange = 0
        self.collided = False

    def collide_to_wall(self):
        self.collided = True
        self.velocity = 0

    def is_collided(self):
        return self.collided

    def updatePos(self):
        self.collisionbox[0] = [self.position - 1, self.position + 1]
        self.position += self.velocity
        self.collisionbox[1] = [self.position - 1, self.position + 1]

    def updateVeloc(self, inputvel):
        temp = self.velocity

        self.velocity = inputvel
        self.velchange = self.velocity - temp


