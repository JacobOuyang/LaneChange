import Cars
import numpy as np
import math
import cv2
import random
class GameV1:
    def __init__(self, render):
        #eachunit is 5.2 feet
        #car will be 3 units long
        #
        self.lanes = 5
        self.Game = []
        self.maxUnits = 500;
        self.playercarposition = 0;
        self.playerlanes = self.lanes-1
        self.imagearray = np.zeros(shape=(200,300))
        self.gameplayerindexvert =0
        self.gameplayerindexhorz =0
        self.render = render
        self.temprestore = []
        self.timer = 0
        self.colorImage = None

    def populateGameArray(self):
        self.Game=[]
        self.timer =0
        import time
        random.seed(int(time.time()))
        for i in range(self.lanes):
            self.Game.append([])
        for i in range(self.lanes):
            currentposition = 0;
            while(currentposition<self.maxUnits):
                self.Game[i].append(Cars.Car(currentposition, np.random.normal(2+(1+0.3*(self.lanes-i-1)), 0.3)))
                currentposition += random.randrange(8,20)
                #currentposition += 10
        self.Game[self.lanes-1][0] = Cars.PlayerCar(0, self.Game[self.lanes-1][1].velocity)

    def updateGameArray(self, action):
        self.timer +=1
        self.tempstore = self.Game
        self.searchforPlayerCar()
        reward= self.updatePlayerCar(action)

        for i in range(len(self.Game)):
            for j in range(len(self.Game[i])):
                updatingcar = self.Game[i][j]
                if (isinstance(updatingcar, Cars.PlayerCar) == False):
                    if j+1 <= len(self.Game[i]) -1:
                        if updatingcar.GetPos() - self.Game[i][j+1].GetPos() <= 0.3:
                            updatingcar.updateVeloc(self.Game[i][j+1].GetVel())
                    if j-1 != 0:
                        if updatingcar.GetVel() < self.Game[i][j-1].GetVel():
                            if self.timer%10 == 0:
                                updatingcar.updateVeloc(np.random.normal(2+(1+0.3*(self.lanes-i-1)), 0.15))
                    updatingcar.updatePos()

                    if updatingcar.GetPos() >= self.maxUnits:
                        self.Game[i].pop(j)
                        self.Game[i].insert(0, updatingcar)
                        updatingcar.position = 0;
        self.searchforPlayerCar()
        updatingcar = self.Game[self.playerlanes][self.playercarposition]
        updatingcar.updatePos()
        #if action ==2 or action ==3:
        #    if self.playercarposition != len(self.Game[self.playerlanes]) -1:
        #        updatingcar.velocity = self.Game[self.playerlanes][self.playercarposition+1].GetVel()
        print('lanes: {}, action: {}, velocity: {}'.format(self.playerlanes, action, updatingcar.GetVel()))

        self.createImage(self.createImageList(), self.gameplayerindexvert, self.gameplayerindexhorz)

        self.render_image()

        if self.checkColission():
            print("crash")
            return -1, 0
        elif updatingcar.GetPos() >=self.maxUnits:
            print("win")
            return 1, reward
        else:
            return 0, reward

    def convert_color_image(self):
        w, h = np.shape(self.imagearray)
        if self.colorImage is None:
            self.colorImage = np.empty((w, h, 3), dtype=np.uint8)
        for i in range(w):
            for j in range(h):
                if self.imagearray[i, j] == 1:
                    self.colorImage[i, j, :] = np.array([10, 200, 200])
                elif self.imagearray[i, j] == 0.5:
                    self.colorImage[i, j, :] = np.array([50, 50, 170])
                else:
                    self.colorImage[i, j, :] = np.array([50, 50, 50])


    def render_image(self):

        if self.render:
            self.convert_color_image()
            #colorImage = cv2.cvtColor(np.float32(self.imagearray), cv2.COLOR_GRAY2BGR)
            #colorImage = create_color_image()
            cv2.imshow('game image', self.colorImage) #self.imagearray)
            cv2.waitKey(100)

    def searchforPlayerCar(self):
        for i in range(len(self.Game)):
            for j in range(len(self.Game[i])):
                if isinstance(self.Game[i][j], Cars.PlayerCar):
                    self.playerlanes = i
                    self.playercarposition =j
                    break


    def updatePlayerCar(self, action):
        #action 0 = velocity of car position +1
        #action 1 = velocity +=2
        #action 2 = left lane change
        #action 3 = right lane change

        playercar = self.Game[self.playerlanes][self.playercarposition]
        if action == 0:

            if self.playercarposition != len(self.Game[self.playerlanes]) -1:
                playercar.updateVeloc(self.Game[self.playerlanes][self.playercarposition+1].GetVel())
                #if playercar.velchange > 0:
                #    return playercar.velchange
                #if playercar.velchange < 0:
                #    return playercar.velchange
                return 0
        elif action == 1:
            #self.searchforPlayerCar()
            playercar.updateVeloc(playercar.GetVel() + 0.1)
            return 0
        elif action == 2:
        #elif action ==1:

            if self.playerlanes != 0:

                for i in range(len(self.Game[self.playerlanes -1])):
                    if self.Game[self.playerlanes-1][i].GetPos() >= playercar.GetPos():
                        self.Game[self.playerlanes].pop(self.playercarposition)
                        self.Game[self.playerlanes-1].insert(i, playercar)
                        return self.Game[self.playerlanes-1][i+1].GetVel() - playercar.GetVel()
            else:
                if self.playercarposition != len(self.Game[self.playerlanes]) - 1:
                    playercar.updateVeloc(self.Game[self.playerlanes][self.playercarposition + 1].GetVel())
        elif action == 3:
        #elif action ==2:

            if self.playerlanes != self.lanes-1:

                for j in range(len(self.Game[self.playerlanes+1])):
                    if self.Game[self.playerlanes+1][j].GetPos() >= playercar.GetPos():
                        self.Game[self.playerlanes].pop(self.playercarposition)
                        self.Game[self.playerlanes+1].insert(j, playercar)
                        return self.Game[self.playerlanes+1][j+1].GetVel() - playercar.GetVel()
            else:
                if self.playercarposition != len(self.Game[self.playerlanes]) -1:
                    playercar.updateVeloc(self.Game[self.playerlanes][self.playercarposition+1].GetVel())

        return 0
    def checkColission(self):
        playercar = self.Game[self.playerlanes][self.playercarposition]
        if self.checkBack(playercar):
            #print("1")
            return True
        elif self.checkFront(playercar):
            #print("2")
            return True
        elif self.checkFast(playercar):
            #print("3")
            return True
        else:
            return False
    def checkBack(self, playercar):
        if self.playercarposition !=0:
            if playercar.CollisionBox()[1][0] <= self.Game[self.playerlanes][self.playercarposition-1].CollisionBox()[1][1]:
                #print(playercar.CollisionBox()[1][0])
                #print("playercar")
                #print(self.Game[self.playerlanes][self.playercarposition-1].CollisionBox()[1][1])
                #print("car behind")
                return True
        return False

    def checkFront(self, playercar):
        if self.playercarposition < (len(self.Game[self.playerlanes])-1):

            if playercar.CollisionBox()[1][1] >= self.Game[self.playerlanes][self.playercarposition+1].CollisionBox()[1][0]:
                #print(playercar.CollisionBox()[1][1])
                #print(self.Game[self.playerlanes][self.playercarposition+1].CollisionBox()[1][0])
                return True

        return False
    def checkFast(self,playercar):
        if (self.playercarposition!=0):

            if playercar.CollisionBox()[0][1] <= self.Game[self.playerlanes][self.playercarposition-1].CollisionBox()[0][0]:

                if playercar.CollisionBox()[1][0] >= self.Game[self.playerlanes][self.playercarposition-1].CollisionBox()[1][1]:
                    #print("2 fast")
                    return True

        return False

    def createImageList(self):
        playercar = self.Game[self.playerlanes][self.playercarposition]
        subGamearray = []
        for i in range(len(self.Game)):
            subGamearray.append([])
        for i in range(len(self.Game)):
            for j in range(len(self.Game[i])):
                if self.Game[i][j].GetPos() >= playercar.GetPos() - 10:
                    if self.Game[i][j].GetPos() <= playercar.GetPos():
                        subGamearray[i].append((self.Game[i][j].GetPos() - playercar.GetPos()) * 10 + 100)
                    if self.Game[i][j].GetPos() == playercar.GetPos():
                        self.gameplayerindexvert = i
                        self.gameplayerindexhorz = len(subGamearray[i]) - 1
                if self.Game[i][j].GetPos() <= playercar.GetPos() + 20:
                    if self.Game[i][j].GetPos() > playercar.GetPos():
                        subGamearray[i].append((self.Game[i][j].GetPos() - playercar.GetPos()) * 10 + 100)
        return subGamearray

    def createImage(self, subGamearray, playervert, playerhorz):
        self.imagearray = np.zeros(shape=(200,300))
        self.imagearray.fill(0.1)
        for i in range(len(subGamearray)):
            for j in range(len(subGamearray[i])):
                if i == playervert:
                    if j == playerhorz:
                        self.shadewhere(subGamearray[i][j], i, 0.5)
                    else:
                        self.shadewhere(subGamearray[i][j], i, 1)
                else:
                    self.shadewhere(subGamearray[i][j], i, 1)
    def shadewhere(self, position, lanenumber, value):
        leftmax = int(math.floor(position - 10))
        rightmax = int(math.floor(position + 10))
        if leftmax < 0:
            leftmax = 0
        if rightmax >300:
            rightmax =300

        for i in range(20):
            for j in range(rightmax-leftmax):
                self.imagearray[i+10 + lanenumber*40][j+leftmax] = value
    def runGame(self, action, greedy):

        #if self.i == 0:
         #   temp = [0,0]
        #else:

        temp = self.updateGameArray(action)
        if greedy == True and temp[0] != 0:
            self.Game=self.temprestore
            return "REDO", 0, 0, 0
        if temp[0] == 0:
            return self.imagearray, temp[0], temp[1], False
            #self.i = 1
        else:
            #self.i=0
            self.populateGameArray()

            return self.imagearray, temp[0], 0, True



def main():
    display = True
    game = GameV1(display)
    game.populateGameArray()
    gameover = False
    cv2.namedWindow("game images")
    while True:
        game.runGame(0, False)
   # while gameover == False:
    #    temp = game.updateGameArray(0)
     #   if temp != None:
      #      gameover = True
       #     print(temp)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


