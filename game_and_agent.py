import numpy as np
import random
from typing import List, Tuple

import pickle


INIT_Q_VALUE = 0

# The 8 symmetries of the board, used by QLearningAgent.get_symmetrical_state_action_pairs().
#
# A rotation or flip only rearranges the 9 cells, and the rearrangement is the same for every board, so it is
# computed once here instead of rotating numpy boards on every call. Cells are numbered row by row,
# cell = 3 * row + col, which is also the order of the characters in a state_key.
#
# Applying a transformation to a grid of cell numbers shows where every cell comes from.
# Example, rotate 90 degrees counter-clockwise:
#     0 1 2        2 5 8
#     3 4 5   ->   1 4 7   ->   flattened: [2, 5, 8, 1, 4, 7, 0, 3, 6]
#     6 7 8        0 3 6
# Read it as "new cell k takes the piece of old cell permutation[k]":
# new cell 0 (top-left) takes old cell 2 (top-right), new cell 1 (top-middle) takes old cell 5 (middle-right), ...
_CELL_GRID = np.arange(9).reshape(3, 3)
SYMMETRY_PERMUTATIONS = [
    transformed.flatten().tolist()
    for transformed in (
        _CELL_GRID,  # identity:    [0, 1, 2, 3, 4, 5, 6, 7, 8]
        np.rot90(_CELL_GRID),  # rotate 90:   [2, 5, 8, 1, 4, 7, 0, 3, 6]
        np.rot90(_CELL_GRID, 2),  # rotate 180:  [8, 7, 6, 5, 4, 3, 2, 1, 0]
        np.rot90(_CELL_GRID, 3),  # rotate 270:  [6, 3, 0, 7, 4, 1, 8, 5, 2]
        np.fliplr(_CELL_GRID),  # flip_lr:     [2, 1, 0, 5, 4, 3, 8, 7, 6]  left-right flip
        np.flipud(_CELL_GRID),  # flip_ud:     [6, 7, 8, 3, 4, 5, 0, 1, 2]  up-down flip
        np.fliplr(np.rot90(_CELL_GRID)),  # flip_45:  [8, 5, 2, 7, 4, 1, 6, 3, 0]  anti-diagonal flip
        np.fliplr(np.rot90(_CELL_GRID, 3)),  # flip_135: [0, 3, 6, 1, 4, 7, 2, 5, 8]  main-diagonal flip
    )
]


def _invert_permutation(permutation: List[int]) -> List[int]:
    """
    Invert a cell permutation.

    If new cell k takes the piece of old cell permutation[k], then a piece in old cell c lands on new cell inverse[c].
    A move needs this direction: we know the move's old cell and want its new cell.

    Example, rotate 90: permutation = [2, 5, 8, 1, 4, 7, 0, 3, 6].
    Old cell 5 (middle-right) is at position 1, so inverse[5] = 1 (top-middle).
    The full inverse is [6, 3, 0, 7, 4, 1, 8, 5, 2].

    Parameters:
    permutation (List[int]): new cell k takes the piece of old cell permutation[k].

    Returns:
    List[int]: inverse[c] is the new cell of old cell c.
    """
    inverse = [0] * len(permutation)
    for new_cell, old_cell in enumerate(permutation):
        inverse[old_cell] = new_cell
    return inverse


INVERSE_SYMMETRY_PERMUTATIONS = [_invert_permutation(p) for p in SYMMETRY_PERMUTATIONS]


class TicTacToe:
    """
    A class for the Tic Tac Toe game.
    """

    def __init__(self) -> None:
        """
        Initializes the board as a 3x3 matrix of zeros.
        """
        self.board = np.zeros((3, 3), dtype=int)
        self.move_record = []

    def is_valid_move(self, x: int, y: int) -> bool:
        """
        Checks if the given move is valid or not.

        Parameters:
        x (int): Row index of the move.
        y (int): Column index of the move.

        Returns:
        bool: True if the move is valid, False otherwise.
        """
        return self.board[x, y] == 0

    def make_move(self, x: int, y: int, player: int) -> bool:
        """
        Updates the board with the given move.

        Parameters:
        x (int): Row index of the move.
        y (int): Column index of the move.
        player (int): The player making the move, can be 1 or 2.

        Returns:
        bool: True if the move is made, False otherwise.
        """
        if self.is_valid_move(x, y):
            self.board[x, y] = player
            self.move_record.append((player, (x, y)))
            return True
        return False

    def withdraw_move(self, x: int, y: int) -> bool:
        """
        Withdraw a move

        Parameters:
        x (int): Row index of the move.
        y (int): Column index of the move.

        Returns:
        bool: True if the move is withdrawn, False otherwise.
        """
        if not self.is_valid_move(x, y):
            self.board[x, y] = 0
            self.move_record.pop()
            return True
        return False

    def check_win(self, player: int) -> bool:
        """
        Checks if the given player has won.

        Parameters:
        player (int): The player who wins is to be checked, can be 1 or 2.

        Returns:
        bool: True if the player has won, False otherwise.
        """
        for row in range(3):
            if np.all(self.board[row, :] == player):
                return True
        for col in range(3):
            if np.all(self.board[:, col] == player):
                return True
        if np.all(np.diag(self.board) == player):
            return True
        if np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def check_draw(self) -> bool:
        """
        Checks if the game is drawn.

        Returns:
        bool: True if the game is drawn, False otherwise.
        """
        return bool(np.all(self.board != 0)) and not (self.check_win(1) or self.check_win(2))

    def reset(self) -> None:
        """
        Resets the board to initial state.
        """
        self.board.fill(0)
        self.move_record = []

    def get_state_key(self) -> str:
        """
        Returns a string representing the current state of the board by flattening the board 2d array.

        Returns:
        str: A string representing the current state of the board.
        """
        return self.board_to_state_key(self.board)

    def get_valid_actions(self) -> list:
        """
        Returns a list of all valid moves.

        Returns:
        list: A list of tuples containing valid move coordinates.
        """
        return [(x, y) for x in range(3) for y in range(3) if self.is_valid_move(x, y)]

    def get_board(self) -> np.ndarray:
        """
        Get the board of the game.

        return:
        np.ndarray: A 3x3 numpy array representing the board.
        """
        return self.board

    def set_board(self, board: np.ndarray) -> None:
        """
        Set the board of the game

        Parameters:
        board (np.ndarray): A 3x3 numpy array representing the board.
        """
        self.board = board

    def set_board_by_state_key(self, state_key: str) -> None:
        """
        Set the board of the game by the state_key string

        Parameters:
        board (str): A state_key string of length 9 representing the board.
        """
        board_array = self.state_key_to_board(state_key)
        self.board = board_array

    @staticmethod
    def state_key_to_board(state_key: str) -> np.ndarray:
        """
        Convert a state_key string to a 3x3 NumPy array representing the TicTacToe board.

        Parameters:
        state_key (str): A string representation of the TicTacToe board, where each character
                         corresponds to the value at a specific board position (1 for 'X', 2 for 'O', 0 for empty).

        Returns:
        np.ndarray: A 3x3 NumPy array representing the TicTacToe board.
        """
        return np.array(list(map(int, state_key))).reshape((3, 3))

    @staticmethod
    def board_to_state_key(board: np.ndarray) -> str:
        """
        Convert a 3x3 NumPy array representing the TicTacToe board to a state_key string.

        Parameters:
        board (np.ndarray): A 3x3 NumPy array representing the TicTacToe board, where each value
                            corresponds to the state at a specific board position (1 for 'X', 2 for 'O', 0 for empty).

        Returns:
        str: A string representation of the TicTacToe board, where each character
             corresponds to the value at a specific board position (1 for 'X', 2 for 'O', 0 for empty).
        """
        return "".join(map(str, board.flatten()))


class QLearningAgent:
    """
    A class for a Q-learning agent that learns to play a game using the Q-learning algorithm.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 1,
        epsilon: float = 0.1,
        pre_trained_q_table: str = "",
    ) -> None:
        """
        Initializes the Q-learning agent.

        Parameters:
        alpha (float): Learning rate (default: 0.1)
        gamma (float): Discount factor (default: 1)
        epsilon (float): Exploration rate (default: 0.1)
        pre_trained_q_table (str): Path to a pre-trained Q-table (default: '')

        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        if pre_trained_q_table:
            with open(pre_trained_q_table, "rb") as file:
                self.q_table = pickle.load(file)
        else:
            self.q_table = {}

    def get_q_value(self, state_key: str, action: Tuple[int, int]) -> float:
        """
        Returns the Q-value for a given state-action pair.

        Parameters:
        state_key (str): A string representing the state of the game.
        action (Tuple[int, int]): A tuple representing the coordinates of the action.

        Returns:
        float: The Q-value for the given state-action pair.
        """
        return self.q_table.get((state_key, action), INIT_Q_VALUE)

    def set_q_value(
        self, state_key: str, action: Tuple[int, int], value: float
    ) -> None:
        """
        Sets the Q-value for a given state-action pair.

        Parameters:
        state_key (str): A string representing the state of the game.
        action (Tuple[int, int]): A tuple representing the coordinates of the action.
        value (float): The new Q-value for the given state-action pair.
        """
        self.q_table[(state_key, action)] = value

    def get_symmetrical_state_action_pairs(
        self, state_key: str, action: Tuple[int, int]
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Get all 8 symmetrical state-action pairs for a given state-action pair.

        Rotating or flipping the board does not change how good a move is, so learn() writes the same Q-value
        to all 8 versions: identity, 3 rotations and 4 flips.

        A symmetry only rearranges the 9 cells, so each pair is built from the precomputed cell lookups
        SYMMETRY_PERMUTATIONS and INVERSE_SYMMETRY_PERMUTATIONS (see the top of this file). Cells are numbered
        row by row, cell = 3 * row + col, which is the order of the characters in state_key.
        - New state_key: new cell k takes the piece of old cell permutation[k].
        - New action: the move's old cell a lands on new cell inverse[a].

        Example: state_key = '010000200', action = (1, 2), rotate 90 degrees counter-clockwise.
            permutation = [2, 5, 8, 1, 4, 7, 0, 3, 6]
            inverse     = [6, 3, 0, 7, 4, 1, 8, 5, 2]

            old board (* = move)        new board
                0 1 0                     0 * 0
                0 0 *        ->           1 0 0
                2 0 0                     0 0 2

            new state_key = state_key[2] + state_key[5] + state_key[8]
                          + state_key[1] + state_key[4] + state_key[7]
                          + state_key[0] + state_key[3] + state_key[6]
                          = '000' + '100' + '002' = '000100002'
            action (1, 2) is cell 3 * 1 + 2 = 5. inverse[5] = 1 (5 is at position 1 of the permutation),
            and divmod(1, 3) = (0, 1), so the new action is (0, 1).

        All 8 pairs for this example, in the order of SYMMETRY_PERMUTATIONS:
            identity     ('010000200', (1, 2))
            rotate 90    ('000100002', (0, 1))
            rotate 180   ('002000010', (1, 0))
            rotate 270   ('200001000', (2, 1))
            flip_lr      ('010000002', (1, 0))
            flip_ud      ('200000010', (1, 2))
            flip_45      ('000001200', (0, 1))
            flip_135     ('002100000', (2, 1))

        Duplicates are not dropped. When the board itself is symmetric some pairs repeat, e.g. the empty board
        with the centre move (1, 1) gives the same pair 8 times. Writing the same value to the same key again
        is harmless, while predicting the number of unique pairs is error-prone: '010000200' with action (1, 2)
        has 8 unique pairs, although its next state '010001200' is symmetric about the anti-diagonal and has
        only 4 unique versions.

        Parameters:
        state_key (str): A string representing the current state of the board.
        action (Tuple[int, int]): a tuple representing the action to be taken.

        Returns:
        List[Tuple[str, Tuple[int, int]]]: A list of 8 symmetrical state-action pairs (with plain int coordinates).
        """
        action_cell = 3 * action[0] + action[1]
        if state_key[action_cell] != "0":
            raise Exception(
                f"The action {action} is not valid for the state key {state_key}."
            )

        symmetrical_state_actions = []
        for permutation, inverse in zip(
            SYMMETRY_PERMUTATIONS, INVERSE_SYMMETRY_PERMUTATIONS
        ):
            # e.g. rotate 90: '010000200' -> '000100002'
            symmetrical_state_key = "".join(state_key[cell] for cell in permutation)
            # e.g. rotate 90: cell 5 = (1, 2) -> cell inverse[5] = 1 = (0, 1)
            symmetrical_action = divmod(inverse[action_cell], 3)
            symmetrical_state_actions.append(
                (symmetrical_state_key, symmetrical_action)
            )

        return symmetrical_state_actions

    def choose_action(
        self,
        state_key: str,
        valid_actions: List[Tuple[int, int]],
        is_learning: bool = True,
    ) -> Tuple[int, int]:
        """
        Chooses an action using the epsilon-greedy strategy.

        Parameters:
        state_key (str): A string representing the state of the game.
        valid_actions (List[Tuple[int, int]]): A list of valid actions.
        is_learning (bool): A flag indicating whether the agent is in learning mode or not (default: True).
                            If not, the agent will not explore.

        Returns:
        Tuple[int, int]: The chosen action.
        """

        if is_learning and random.random() < self.epsilon:
            return random.choice(valid_actions)

        # all the symmetric states is actually one state. We don't need to choose the max_q_value among the symmetric
        # states, because they should share the same q_values.

        q_values = [self.get_q_value(state_key, action) for action in valid_actions]

        max_q_value = -np.inf
        best_actions = []
        for action, q_value in zip(valid_actions, q_values):
            if q_value > max_q_value:
                max_q_value = q_value
                best_actions = [action]
            elif q_value == max_q_value:
                best_actions.append(action)
        return random.choice(best_actions)

    def learn(
        self,
        state_key: str,
        action: Tuple[int, int],
        reward: float,
        next_state_key: str,
        next_valid_actions: List[Tuple[int, int]],
    ) -> None:
        """
        Updates the Q-table using the Q-learning update rule.

        Parameters:
        state_key (str): A string representing the state of the game before taking the action.
        action (Tuple[int, int]): A tuple representing the coordinates of the action.
        reward (float): The reward received for taking the action.
        next_state_key (str): A string representing the state of the game after taking the action and the opponent's move.
        next_valid_actions (List[Tuple[int, int]]): A list of valid actions in the next state.

        Note: the next_valid_actions is redundant. It can be derived from next_state_key, but it is convenient to pass it.
        And it serves as a flag for the terminal state.
        """

        current_q_value = self.get_q_value(state_key, action)

        # Update the Q-table using the Q-learning update rule
        # The usual update rule is
        # Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') - Q(s, a)]

        # If state_key is the terminal state (win, lose, or draw), the next_valid_actions will be [],
        # so the next_max_q_value is 0 (by definition of the terminal states values). Then, We update the Q value by
        # Q(s, a) ← Q(s, a) + α[r - Q(s, a)]
        next_max_q_value = (
            max(
                [
                    self.get_q_value(next_state_key, next_action)
                    for next_action in next_valid_actions
                ]
            )
            if next_valid_actions
            else 0
        )

        new_q_value = current_q_value + self.alpha * (
            reward + self.gamma * next_max_q_value - current_q_value
        )

        symmetrical_states_and_actions = self.get_symmetrical_state_action_pairs(
            state_key, action
        )

        for sym_state_key, sym_action in symmetrical_states_and_actions:
            self.set_q_value(sym_state_key, sym_action, new_q_value)
