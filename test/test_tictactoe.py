import json

from game_and_agent import QLearningAgent, TicTacToe, make_opponent_move
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

    def test_save_and_load_q_table(self, tmp_path):
        agent = QLearningAgent()
        agent.set_q_value("000000000", (0, 0), 0.7818519084051515)
        agent.set_q_value("000000000", (1, 1), -0.1)
        agent.set_q_value("010000200", (1, 2), 0.25)
        path = tmp_path / "q_table.json"
        agent.save_q_table(path)

        loaded_agent = QLearningAgent(pre_trained_q_table=path)

        assert (
            loaded_agent.q_table == agent.q_table
            and all(type(row) is int and type(col) is int for _, (row, col) in loaded_agent.q_table)
        )

    def test_saved_q_table_format(self, tmp_path):
        agent = QLearningAgent()
        agent.set_q_value("000000000", (0, 1), 0.5)
        path = tmp_path / "q_table.json"
        agent.save_q_table(path)

        assert json.loads(path.read_text()) == {"000000000": {"0,1": 0.5}}

    def test_load_pre_trained_q_tables(self):
        # The exact number of entries changes whenever the agents are retrained, so check the shape instead.
        for path in ("q_table_ubuntu_agent_move_first.json", "q_table_ubuntu_agent_move_second.json"):
            q_table = QLearningAgent(pre_trained_q_table=path).q_table

            assert q_table and all(
                len(state_key) == 9
                and set(state_key) <= set("012")
                and type(row) is int
                and type(col) is int
                and state_key[3 * row + col] == "0"
                and isinstance(value, float)
                for (state_key, (row, col)), value in q_table.items()
            )

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
        result = game.withdraw_move()

        assert (
            result
            and TicTacToe.board_to_state_key(game.board) == "212000000"
            and game.move_record == []
        )

    def test_withdraw_move_only_undoes_the_last_move(self):
        game = TicTacToe()
        game.make_move(0, 0, 1)
        game.make_move(1, 1, 2)
        result = game.withdraw_move()

        assert (
            result
            and TicTacToe.board_to_state_key(game.board) == "100000000"
            and game.move_record == [(1, (0, 0))]
        )

    def test_withdraw_move_without_a_move_to_withdraw(self):
        state_key = "212000000"
        game = TicTacToe()
        game.set_board_by_state_key(state_key)
        result = game.withdraw_move()

        assert (
            not result
            and TicTacToe.board_to_state_key(game.board) == "212000000"
            and game.move_record == []
        )

    def test_set_board_clears_move_record(self):
        game = TicTacToe()
        game.make_move(0, 0, 1)
        game.set_board(TicTacToe.state_key_to_board("000010000"))
        record_after_set_board = list(game.move_record)

        game.make_move(0, 0, 1)
        game.set_board_by_state_key("000010000")

        assert record_after_set_board == [] and game.move_record == []

    def test_state_key_to_board(self):
        state_key = "200000000"

        result = TicTacToe.state_key_to_board(state_key)

        assert (result == np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]])).all()

    def test_find_winning_move(self):
        state_key = "220110000"
        game = TicTacToe()
        game.set_board_by_state_key(state_key)

        assert (
            game.find_winning_move(2) == (0, 2)
            and game.find_winning_move(1) == (1, 2)
            and TicTacToe.board_to_state_key(game.board) == state_key
            and game.move_record == []
        )

    def test_find_winning_move_without_a_win(self):
        game = TicTacToe()
        game.set_board_by_state_key("200010000")

        assert game.find_winning_move(2) is None and game.find_winning_move(1) is None


class TestMakeOpponentMove:
    def test_opponent_plays_a_winning_move(self):
        game = TicTacToe()
        game.set_board_by_state_key("220110000")
        action = make_opponent_move(game)

        assert action == (0, 2) and game.check_win(2)

    def test_random_opponent_plays_a_valid_move(self):
        state_key = "200010000"
        game = TicTacToe()
        game.set_board_by_state_key(state_key)
        action = make_opponent_move(game)

        assert (
            state_key[3 * action[0] + action[1]] == "0"
            and game.board[action] == 2
            and game.move_record == [(2, action)]
        )

    def test_agent_opponent_sees_the_board_with_players_swapped(self):
        class RecordingAgent:
            def choose_action(self, state_key, valid_actions, is_learning=True):
                self.seen = (state_key, is_learning)
                return valid_actions[0]

        game = TicTacToe()
        game.set_board_by_state_key("100000000")
        agent = RecordingAgent()
        action = make_opponent_move(game, agent, is_learning=False)

        assert agent.seen == ("200000000", False) and action == (0, 1) and game.board[0, 1] == 2
