import numpy as np
row,col = 10,5 #board size

class AI():
    def __init__(self):
     
        self.gameover = False
        self.board = np.zeros([row,col], dtype=int)
        self.stable = True #False:need to clean
        self.step = 0 #total turns , start from 0
        self.turn = -1 
        self.mypoints = 0
        self.oppopoints = 0
        self.board = np.loadtxt("board2.txt", dtype= int)
        
        # #make board
        # tmp = (np.arange(row)/2)+1
        # while(1):
        #     for i in range(col):
        #         np.random.shuffle(tmp)
        #         self.board[:,i] = tmp
        #     if self.checkstable() == True:
        #         break
        # self.show_board()
        
    def checkstable(self):
        for r in range(len(self.board)):
            for c in range(len(self.board[0])-2):
                if abs(self.board[r][c]) == abs(self.board[r][c+1]) == abs(self.board[r][c+2]) != 0:
                    self.stable = False
                    return self.stable
        self.stable = True
        return self.stable

    def drop(self): # drop the tile 
        for c in range(col):
            if self.board[:,c].sum()!=0:
                k = -len(self.board[:,c][self.board[:,c]>0])
                self.board[k:,c] = self.board[:,c][self.board[:,c]>0]
                self.board[:k,c] = 0

    def checkgameover(self): #any of last row == 0
        self.gameover = np.any(self.board[-1] == 0)
        return self.gameover

    def clean(self): #clean the tile , return points
        unstable = 1
        points = 0
        while(unstable):
            unstable = 0
            for i in range(row):
                for j in range(col-2):
                    if abs(self.board[i][j]) == abs(self.board[i][j+1]) == abs(self.board[i][j+2]) != 0:
                        self.board[i][j] = self.board[i][j+1] = self.board[i][j+2] = -abs(self.board[i][j])
                        unstable = 1
            points -= self.board[self.board<0].sum()
            self.board[self.board<0] = 0
            self.drop()
        self.checkgameover()        
        return points

    def make_decision(self):
        #######################################################
        ##### This is the main part you need to implement #####
        #######################################################        
        p = 0
        while not(p):
            x= np.random.randint(row)
            y= np.random.randint(col)
            p = self.board[x][y]
        return [x,y]
        # return format : [x,y]
        # Use AI to make decision !
        # random is only for testing !
    
    def rand_select(self):
        p = 0
        while not(p):
            x= np.random.randint(row)
            y= np.random.randint(col)
            p = self.board[x][y]
        return [x,y]

    def make_move(self, x, y):
        pts = self.board[x][y]
        self.board[1:x+1,y] = self.board[0:x,y]
        self.board[0][y] = 0

        if self.checkgameover(): 
            return pts
        
        pts += self.clean()
        return pts
        
    def start(self):
        print("Game start!")      
        print('――――――――――――――――――')
        self.show_board()
        self.turn = int(input("Set the player's order(0:first, 1:second): "))


        #start playing    
        while not self.gameover:
            print('Turn:', self.step)
            if ((self.step%4)//2) == self.turn:
                print('It\'s your turn')
                x,y = self.make_decision()
                print(f"Your move is {x},{y}.")
                # [x,y] = [int(x) for x in input("Enter the move : ").split()]
                assert (0<=x and x<=row-1 and 0<=y and y<=col-1)
                assert (self.board[x][y]>0)
                pts = self.make_move(x,y)
                self.mypoints += pts
                print(f'You get {pts} points')  
                self.show_board()

            else:
                print('It\'s opponent\'s turn')
                # x,y = self.rand_select() # can use this while testing ,close it when you submit
                [x,y] = [int(x) for x in input("Enter the move : ").split()] #open it when you submit
                assert (0<=x and x<=row-1 and 0<=y and y<=col-1)
                assert (self.board[x][y]>0)
                print(f"Your opponent move is {x},{y}.")
                pts = self.make_move(x,y)
                self.oppopoints += pts
                print(f'Your opponent\'s get {pts} points')
                self.show_board()

            self.step += 1

        #gameover
        if self.mypoints > self.oppopoints:
            print('You win!')
            return 1
        elif self.mypoints < self.oppopoints:
            print('You lose!')
            return -1
        else:
            print('Tie!')
            return 0

    def show_board(self):
        print('my points:', self.mypoints)
        print('opponent\'s points:', self.oppopoints)
        print('The board is :')
        print(self.board)
        print('――――――――――――――――――')

if __name__ == '__main__':
 
    game = AI()
    game.start()
