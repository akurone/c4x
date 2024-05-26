from env import Env
from abc import ABC, abstractmethod
from board import ConnectFourField
from copy import deepcopy
import random
from env import  Env, FIELD_COLUMNS, FIELD_ROWS
import torch
import random
import numpy as np

RTG = 2.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_DIM = FIELD_COLUMNS * FIELD_ROWS
ACT_DIM = FIELD_COLUMNS
MAX_EP = 21

class Agent(ABC):
    def __init__(self, player: int):
        self.player = player

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self, env: Env, **kwargs):
        pass

class RandomAgent(Agent):
    """
    Implementation of the agent following random policy, not learning at all. 
    """
    def __init__(self, player: int):
        super(RandomAgent, self).__init__(player)

    def reset(self):
        pass

    def act(self, env: Env, **kwargs):
        return env.random_valid_action()

class MinimaxAgent(Agent):
    """
    Implementation of the minimax agent.
    Adapted from https://github.com/AbdallahReda/Connect4/blob/master/minimaxAlphaBeta.py 
    """
    def __init__(self, depth: int = 3, epsilon: float = 0.1, player: int = 2):
        super(MinimaxAgent, self).__init__(player)
        
        self.max_depth = depth
        self.player = player

        # Measure of randomness
        self.epsilon = epsilon

    def reset(self):
        pass

    def act(self, env: Env, **kwargs):
        # Choose best predicted action
        if random.random() > self.epsilon:
            return self.best_predicted_action(env.get_state(return_type="board"), self.max_depth, self.player)
        else:
            return env.random_valid_action()
        
    def decay_epsilon(self, rate: float = 0.9, min: float = 0.1):
        self.epsilon = max(min, self.epsilon * rate)
    
    def set_epsilon(self, value: float = 0.4):
        assert value >= 0.0 and value <= 1.0
        self.epsilon = value

    # Starting from the middle row and going outwards from there can decrease search times by a factor of over 10 
    # as the middle is in general the better column to play
    def reorder_array(self, arr):
        n = len(arr)
        middle = n // 2  # Get the index of the middle element
        reordered = [arr[middle]]  # Start with the middle element

        for i in range(1, middle+1):
            # Add the element to the right and then to the left of the middle, if they exist
            if middle + i < n:
                reordered.append(arr[middle + i])
            if middle - i >= 0:
                reordered.append(arr[middle - i])

        return reordered

    def best_predicted_action(self, board: ConnectFourField, depth: int = 4, player: int = 1):
        # Get array of possible moves
        validMoves = board.get_valid_cols()
        # Choose random starting move
        # shuffle(validMoves)
        validMoves = self.reorder_array(validMoves)

        bestMove  = validMoves[0]
        bestScore = float("-inf")

        # Initial alpha & beta values for alpha-beta pruning
        alpha = float("-inf")
        beta = float("inf")

        if player == 2: opponent = 1
        else: opponent = 2
    
        # Go through all of those moves
        for move in validMoves:
            # Create copy so as not to change the original board
            tempBoard = deepcopy(board)
            tempBoard.play(player, move)

            # Call min on that new board
            boardScore = self.minimizeBeta(tempBoard, depth - 1, alpha, beta, player, opponent)
            if boardScore > bestScore:
                bestScore = boardScore
                bestMove = move
        
        return bestMove

    def minimizeBeta(self, board: ConnectFourField, depth: int, a, b, player: int, opponent: int):
        # Get all valid moves
        validMoves = board.get_valid_cols()
        
        # RETURN CONDITION
        # Check to see if game over
        if depth == 0 or len(validMoves) == 0 or board.is_finished() != -1:
            return board.utilityValue(player)
        
        # CONTINUE TREE SEARCH
        beta = b
        # If end of tree evaluate scores
        for move in validMoves:
            boardScore = float("inf")
            # Else continue down tree as long as ab conditions met
            if a < beta:
                tempBoard = deepcopy(board)
                tempBoard.play(opponent, move)
                boardScore = self.maximizeAlpha(tempBoard, depth - 1, a, beta, player, opponent)
            if boardScore < beta:
                beta = boardScore
        return beta

    def maximizeAlpha(self, board: ConnectFourField, depth: int, a, b, player: int, opponent: int):
        # Get all valid moves
        validMoves = board.get_valid_cols()

        # RETURN CONDITION
        # Check to see if game over
        if depth == 0 or len(validMoves) == 0 or board.is_finished() != -1:
            return board.utilityValue(player)

        # CONTINUE TREE SEARCH
        alpha = a        
        # If end of tree, evaluate scores
        for move in validMoves:
            boardScore = float("-inf")
            # Else continue down tree as long as ab conditions met
            if alpha < b:
                tempBoard = deepcopy(board)
                tempBoard.play(player, move)
                boardScore = self.minimizeBeta(tempBoard, depth - 1, alpha, b, player, opponent)
            if boardScore > alpha:
                alpha = boardScore
        return alpha

class ModelAgent(Agent):
    def __init__(self, model, player: int):
        super(ModelAgent, self).__init__(player)
        self.model = model
        self.reset()

    def reset(self):
        self.states = None
        self.actions = torch.zeros((0, ACT_DIM), device=DEVICE, dtype=torch.float32)
        self.timestep = 0
        self.timesteps = torch.tensor(0, device=DEVICE, dtype=torch.long).reshape(1, 1)
        self.returns_to_go = torch.tensor(RTG, device=DEVICE, dtype=torch.float32).reshape(1, 1)


    def act(self, env: Env, **kwargs):
        return self.get_action(self.model, env, **kwargs)

    def append(self, action, isExplore=False):
        if isExplore:
          onehot = [0,0,0,0,0,0,0]; onehot[action] = 1
          self.actions[-1] = torch.tensor(onehot)
        else:
           self.actions[-1] = action
           action = action.detach().cpu().numpy()
           action = int(np.argmax(action))

        self.timesteps = torch.cat([self.timesteps, torch.ones((1, 1), device=DEVICE, dtype=torch.long) * (self.timestep + 1)], dim=1)
        self.timestep += 1
        return action

    def get_action(self, model, env, **kwargs):
        reward = kwargs["reward"]

        if reward is not None:
            pred_return = self.returns_to_go[0, -1] - reward
            self.returns_to_go = torch.cat([self.returns_to_go, pred_return.reshape(1, 1)], dim=1)

        if self.states is None:
            self.states = torch.from_numpy(np.array(env.get_state()).flatten()).reshape(1, STATE_DIM).to(device=DEVICE, dtype=torch.float32)
        else:
            self.states = torch.cat([self.states, torch.from_numpy(np.array(env.get_state()).flatten()).to(device=DEVICE).reshape(1, STATE_DIM)], dim=0)

        self.actions = torch.cat([self.actions, torch.zeros((1, ACT_DIM), device=DEVICE)], dim=0)

        if "explore" in kwargs and random.random() < kwargs["explore"]:
            return self.append(env.random_valid_action(), True)

        states = self.states.reshape(1, -1, STATE_DIM)[:, -MAX_EP :]
        actions = self.actions.reshape(1, -1, ACT_DIM)[:, -MAX_EP :]
        returns_to_go = self.returns_to_go.reshape(1, -1, 1)[:, -MAX_EP :]
        timesteps = self.timesteps[:, -MAX_EP :]
        padding = MAX_EP - states.shape[1]
        # pad all tokens to sequence length
        attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
        attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
        states = torch.cat([torch.zeros((1, padding, STATE_DIM)), states], dim=1).float()
        actions = torch.cat([torch.zeros((1, padding, ACT_DIM)), actions], dim=1).float()
        returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
        timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)
        action_preds = model.original_forward(
            states=states,
            actions=actions,
            #rewards=rewards,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )
        return self.append(action_preds[0, -1])