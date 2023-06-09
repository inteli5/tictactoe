import numpy as np
import random
from typing import List, Tuple

import pickle


INIT_Q_VALUE = 0


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
        return np.all(self.board != 0)

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
        Get all symmetrical state-action paris for a given state-action pair.

        In before, we want to return non-duplicate state-action pair. We used the next_state_key to decide how many
        non-duplicate state-action pair. But it still has problem.
        For example,
        state_key = '010000200'
        action = (1, 2)
        The new_state_key '010001200' has axillary diagonal symmetry but the state_key does not have any symmetry.
        If we use the next_state_key to decide how many non-duplicate state-action pair, the answer is 4.
        But the correct answer is 8, rather than 4.
        Our algorithm to drop duplicate seems expensive.
        So we choose not to drop duplicates, which should be more efficient.

        Parameters:
        state_key (str): A string representing the current state of the board.
        action (Tuple[int, int]): a tuple representing the action to be taken.

        Returns:
        List[Tuple[str, Tuple[int, int]]]: A list of symmetrical state-action pair.
        """

        # Initialize a temp game object
        game = TicTacToe()
        game.set_board_by_state_key(state_key)
        valid_move = game.make_move(*action, 1)
        if not valid_move:  # AI agent always use 1 'X'.
            raise Exception(
                f"The action {action} is not valid for the state key {state_key}."
            )

        # Convert the state_key to a 3x3 NumPy array
        board = game.board

        # Generate all possible symmetrical states and their transformations
        symmetrical_states_for_next_state_key = {
            ("identity", 1): board,
            # [[0 1 2]
            # [3 4 5]
            # [6 7 8]]
            ("rotate", 1): np.rot90(board),  # 90 degrees counter-clockwise rotation
            # e.g.
            # [[2 5 8]
            # [1 4 7]
            # [0 3 6]]
            ("rotate", 2): np.rot90(board, 2),  # 180 degrees counter-clockwise rotation
            ("rotate", 3): np.rot90(board, 3),  # 270 degrees counter-clockwise rotation
            ("flip_lr", 1): np.fliplr(board),  # Left-right flip
            ("flip_ud", 1): np.flipud(board),  # Up-down flip
            ("flip_45", 1): np.fliplr(
                np.rot90(board)
            ),  # Flip about the 45-degree line (axillary diagonal axis of the matrix)
            # e.g.
            # [[8 5 2]
            #  [7 4 1]
            #  [6 3 0]]
            ("flip_135", 1): np.fliplr(
                np.rot90(board, 3)
            )  # Flip about the 135-degree line (main diagonal axis of the matrix)
            # e.g.
            #  [[0 3 6]
            #  [1 4 7]
            #  [2 5 8]]
        }

        board = TicTacToe.state_key_to_board(state_key)

        symmetrical_states_for_state_key = {
            ("identity", 1): board,
            ("rotate", 1): np.rot90(board),  # 90 degrees counter-clockwise rotation
            ("rotate", 2): np.rot90(board, 2),  # 180 degrees counter-clockwise rotation
            ("rotate", 3): np.rot90(board, 3),  # 270 degrees counter-clockwise rotation
            ("flip_lr", 1): np.fliplr(board),  # Left-right flip
            ("flip_ud", 1): np.flipud(board),  # Up-down flip
            ("flip_45", 1): np.fliplr(
                np.rot90(board)
            ),  # Flip about the 45-degree line (axillary diagonal axis of the matrix)
            ("flip_135", 1): np.fliplr(
                np.rot90(board, 3)
            ),  # Flip about the 135-degree line (main diagonal axis of the matrix)
        }

        symmetrical_state_actions = []
        for transformation in symmetrical_states_for_state_key.keys():
            this_step_board = symmetrical_states_for_state_key[transformation]
            next_step_board = symmetrical_states_for_next_state_key[transformation]

            move = next_step_board - this_step_board
            row, col = np.nonzero(move)
            symmetrical_action = (row[0], col[0])
            symmetrical_state_actions.append(
                (
                    TicTacToe.board_to_state_key(
                        symmetrical_states_for_state_key[transformation]
                    ),
                    symmetrical_action,
                )
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
