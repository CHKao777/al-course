import numpy as np
import time
row,col = 6,3 #board size

class AI():
    def __init__(self):
     
        self.gameover = False
        self.board = np.zeros([row,col], dtype=int)
        self.stable = True #False:need to clean
        self.step = 0 #total turns , start from 0
        self.turn = -1 
        self.mypoints = 0
        self.oppopoints = 0
        self.board = np.loadtxt("board.txt", dtype= int)
        
        #make board
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
        # p = 0
        # while not(p):
        #     x= np.random.randint(row)
        #     y= np.random.randint(col)
        #     p = self.board[x][y]
        # return [x,y]

        #my implementation
        root_state = CantrisGameState(self.board, self.mypoints - self.oppopoints)
        root_node = TwoPlayersGameMonteCarloTreeSearchNode(root_state)
        x, y = MonteCarloTreeSearch(root_node).best_action(total_simulation_seconds=25)
        return [x, y]

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
            if (self.step%2) == self.turn:
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

class CantrisGameState():

    def __init__(self, state, score, action=None, next_to_move=1):
        self.board = state
        self.action = action #take which action to get in this state
        self.next_to_move = next_to_move
        self.score = score

    @property
    def game_result(self):
        # check if game is over
        if not np.any(self.board[-1] == 0):
            return None
        if self.score > 0:
            return 1
        if self.score < 0:
            return 2
        return 0

    def is_game_over(self):
        return self.game_result is not None

    def is_move_legal(self, move):
        x, y = move
        assert (0 <= x and x <= row - 1 and 0 <= y and y <= col - 1)
        assert (self.board[x][y] > 0)

    def move(self, move):
        self.is_move_legal(move)

        x, y = move
        pts = self.board[x][y]
        new_board = np.copy(self.board)
        new_board[1:x+1,y] = new_board[0:x,y]
        new_board[0][y] = 0

        if not np.any(self.board[-1] == 0):
            pts += self.clean(new_board)
        
        score = self.score
        if self.next_to_move == 1:
            score += pts
        else:
            score -= pts
      
        return CantrisGameState(new_board, score, move, 3 - self.next_to_move)

    def clean(self, board): #clean the tile , return points
        unstable = 1
        points = 0
        while (unstable):
            unstable = 0
            for i in range(row):
                for j in range(col-2):
                    if abs(board[i][j]) == abs(board[i][j+1]) == abs(board[i][j+2]) != 0:
                        board[i][j] = board[i][j+1] = board[i][j+2] = -abs(board[i][j])
                        unstable = 1
            points -= board[board<0].sum()
            board[board<0] = 0
            self.drop(board)       
        return points

    def drop(self, board): # drop the tile 
        for c in range(col):
            if board[:,c].sum()!=0:
                k = -len(board[:,c][board[:,c]>0])
                board[k:,c] = board[:,c][board[:,c]>0]
                board[:k,c] = 0

    def get_legal_actions(self):
        indices = np.where(self.board != 0)
        return list(zip(indices[0], indices[1]))


class TwoPlayersGameMonteCarloTreeSearchNode():

    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.n = 0. # num of visits
        self.win = 0. # num of wins
        self.lose = 0. # num of loses
        self._untried_actions = None

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = TwoPlayersGameMonteCarloTreeSearchNode(
            next_state, parent=self
        )
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result

    def backpropagate(self, result):
        self.n += 1.

        if self.parent is not None:
            if self.parent.state.next_to_move == result:
                self.win += 1.
            elif result != 0:
                self.lose += 1.

        if self.parent:
            self.parent.backpropagate(result)
    
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.):
        # print(len(self.children))
        choices_weights = [
            ((c.win - c.lose) / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            # ((c.win) / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):        
        return possible_moves[np.random.randint(len(possible_moves))]


class MonteCarloTreeSearch(object):

    def __init__(self, node):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """
        self.root = node

    def best_action(self, simulations_number=None, total_simulation_seconds=None):
        """

        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action

        total_simulation_seconds : float
            Amount of time the algorithm has to run. Specified in seconds

        Returns
        -------

        """

        if simulations_number is None :
            assert(total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while time.time() < end_time:
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
        else :
            for _ in range(0, simulations_number):            
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
        # to select best child go for exploitation only
            
        return self.root.best_child(c_param=0.).state.action

    def _tree_policy(self):
        """
        selects node to run rollout/playout for

        Returns
        -------

        """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child(c_param=1.)
        return current_node

if __name__ == '__main__':
 
    game = AI()
    game.start()