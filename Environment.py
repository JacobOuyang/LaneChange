import Cars

class GameV1:
    def __init__(self, lanes):
        #eachunit is 5.2 feet
        #car will be 3 units long
        #
        self.lanes = lanes
        self.Game = []
        self.maxUnits = 1000;
        self.playercarposition = 0;
        self.playerlanes =lanes-1


        for i in range(self.lanes):
            self.Game.append([])

    def populateGameArray(self):
        for i in range(self.lanes):
            currentposition = 0;
            while(currentposition<1000):
                self.Game[i].append(Cars.Car(currentposition, 2+(1+0.1*(self.lanes-i))))
                currentposition += 6
        self.Game[self.lanes-1][0] = Cars.PlayerCar(0, self.Game[self.lanes-1][1].velocity)

    def updateGameArray(self, action):
        self.updatePlayerCar(action)

        for i in range(len(self.Game)):
            for j in range(len(self.Game[i])):
                updatingcar = self.Game[i][j]
                if (isinstance(updatingcar, Cars.PlayerCar) == False):
                    updatingcar.updatePos()
                    if (updatingcar.position > 1000):
                        self.Game[i].pop(j)
                        self.Game[i].insert(0, updatingcar)
                        updatingcar.position = 0;
        updatingcar = self.Game[self.playerlanes][self.playercarposition]
        updatingcar.updatePos()
        print(updatingcar.GetPos())
        if self.CheckColission(self.playerlanes, self.playercarposition):
            return "Collision"
        if updatingcar.GetPos() >=1000:
            return "Game Won"


    def updatePlayerCar(self, action):
        #action 0 = velocity of car position +1
        #action 1 = velocity +=2
        #action 2 = left lane change
        #action 3 = right lane change
        playercar = self.Game[self.playerlanes][self.playercarposition]
        if action ==0:
            if self.playercarposition != len(self.Game[self.playerlanes]) -1:
                playercar.updateVeloc(self.Game[self.playerlanes][self.playercarposition+1].GetVel())

        elif action ==1:
            playercar.updateVeloc(playercar.GetVel() + 2)

        elif action ==2:

            if self.playerlanes != 0:
                for i in range(len(self.Game[lane-1])):
                    if self.Game[self.playerlanes-1][i].GetPos() >= playercar.GetPos():
                        self.Game[self.playerlanes].pop(self.playercarposition)
                        self.Game[self.playerlanes-1].insert(i, playercar)


        elif action ==3:
            if self.playerlanes != self.lanes-1:
                for i in range(len(self.Game[self.playerlanes+1])):
                    if self.Game[self.playerlanes+1][i].GetPos >= playercar.GetPos():
                        self.Game.pop([self.playerlanes][self.playercarposition])
                        self.Game[self.playerlanes+1].insert(i, playercar)

    def CheckColission(self, lane, carnumber):
        if carnumber != (len(self.Game[lane]) -1):
            if not set(self.Game[lane][carnumber].CollisionBox()).isdisjoint(self.Game[lane][carnumber+1].CollisionBox()):
                return True
        if carnumber != 0:
            if not set(self.Game[lane][carnumber].CollisionBox()).isdisjoint(self.Game[lane][carnumber-1].CollisionBox()):
                return True
        else:
            return False








def main():
    game = GameV1(5)
    game.populateGameArray()
    gameover = False
    while gameover == False:
        temp = game.updateGameArray(0)
        if temp != None:
            gameover = True
            print(temp)



if __name__ == "__main__":
    main()


