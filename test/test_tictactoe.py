from game_and_agent import QLearningAgent, TicTacToe
import numpy as np

state_key = "200000000"
action = (0, 1)
agent = QLearningAgent()
result = agent.get_symmetrical_state_action_pairs(state_key, action)


class TestQLearningAgent:
    def test_get_symmetrical_state_action_pairs1(self):
        state_key = '010000200'
        action = (1, 2)
        agent = QLearningAgent()
        result = agent.get_symmetrical_state_action_pairs(state_key, action)

        assert set(result) == {
             ('010000200', (1, 2)),
             ('000100002', (0, 1)),
             ('002000010', (1, 0)),
             ('200001000', (2, 1)),
             ('010000002', (1, 0)),
             ('200000010', (1, 2)),
             ('000001200', (0, 1)),
             ('002100000', (2, 1))
        }

class TestGame:
    def test_move_record(self):
        state_key = "212000000"
        action = (1, 1)
        game = TicTacToe()
        game.set_board_by_state_key(state_key)
        game.make_move(*action, 1)

        action = (2, 1)
        result = game.make_move(*action, 2)

        assert (
            result
            and TicTacToe.board_to_state_key(game.board) == "212010020"
            and game.move_record == [(1, (1, 1)), (2, (2, 1))]
        )

    def test_make_valid_move(self):
        state_key = "212000000"
        action = (1, 1)
        game = TicTacToe()
        game.set_board_by_state_key(state_key)
        result = game.make_move(*action, 1)

        assert (
            result
            and TicTacToe.board_to_state_key(game.board) == "212010000"
            and game.move_record == [(1, (1, 1))]
        )

    def test_make_invalid_move(self):
        state_key = "212000000"
        action = (0, 1)
        game = TicTacToe()
        game.set_board_by_state_key(state_key)
        result = game.make_move(*action, 1)

        assert (
            not result
            and TicTacToe.board_to_state_key(game.board) == "212000000"
            and game.move_record == []
        )

    def test_make_move_and_withdraw_move(self):
        state_key = "212000000"
        action = (1, 1)
        game = TicTacToe()
        game.set_board_by_state_key(state_key)
        game.make_move(*action, 1)
        result = game.withdraw_move(*action)

        assert (
            result
            and TicTacToe.board_to_state_key(game.board) == "212000000"
            and game.move_record == []
        )

    def test_state_key_to_board(self):
        state_key = "200000000"

        result = TicTacToe.state_key_to_board(state_key)

        assert (result == np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]])).all()
