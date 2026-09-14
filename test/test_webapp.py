from fastapi.testclient import TestClient

import tictactoe_webapp
from game_and_agent import TicTacToe
from tictactoe_webapp import app

client = TestClient(app)

CORNERS = [(0, 0), (0, 2), (2, 0), (2, 2)]


def post_move(board, player_who_move_first, x, y, difficulty="easy"):
    state = {"board": board, "player_who_move_first": player_who_move_first, "message": "", "difficulty": difficulty}
    return client.post("/make_move", json={"state": state, "x": x, "y": y})


class StubAgent:
    def choose_action(self, state_key, valid_actions, is_learning=True):
        return ("learned move", is_learning)


class TestWebApp:
    def test_home_page_loads(self):
        response = client.get("/")

        assert response.status_code == 200 and '<div class="grid">' in response.text

    def test_avatar_image_is_served(self):
        response = client.get("/static/girl.jpg")

        assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"

    def test_page_uses_relative_urls(self):
        # The app is served under https://lamian.ai/tictactoe/, so the page must not request URLs from the site root.
        page = client.get("/").text

        assert (
            "fetch('new_game'" in page
            and "fetch('make_move'" in page
            and 'src="static/girl.jpg"' in page
            and "fetch('/" not in page
            and 'src="/' not in page
        )

    def test_new_game_ai_first_opens_with_learned_corner_move(self):
        response = client.post("/new_game", json={"player_who_move_first": "X", "difficulty": "hard"})
        board = response.json()["board"]
        x_cells = [(x, y) for x in range(3) for y in range(3) if board[x][y] == "X"]
        o_count = sum(row.count("O") for row in board)

        assert (
            response.status_code == 200
            and len(x_cells) == 1
            and x_cells[0] in CORNERS
            and o_count == 0
        )

    def test_new_game_human_first_is_empty(self):
        response = client.post("/new_game", json={"player_who_move_first": "O"})

        assert (
            response.status_code == 200
            and response.json()["board"] == [["", "", ""], ["", "", ""], ["", "", ""]]
        )

    def test_difficulty_defaults_to_easy(self):
        response = client.post("/new_game", json={"player_who_move_first": "O"})

        assert response.json()["difficulty"] == "easy"

    def test_new_game_keeps_difficulty(self):
        response = client.post("/new_game", json={"player_who_move_first": "O", "difficulty": "medium"})

        assert response.status_code == 200 and response.json()["difficulty"] == "medium"

    def test_unknown_difficulty_is_rejected(self):
        response = client.post("/new_game", json={"player_who_move_first": "O", "difficulty": "impossible"})

        assert response.status_code == 422

    def test_difficulty_is_used_for_ai_moves(self, monkeypatch):
        used = []

        def record_difficulty(agent, game, difficulty):
            used.append(difficulty)
            return game.get_valid_actions()[0]

        monkeypatch.setattr(tictactoe_webapp, "choose_agent_move", record_difficulty)
        client.post("/new_game", json={"player_who_move_first": "X", "difficulty": "medium"})
        post_move([["X", "", ""], ["", "", ""], ["", "", ""]], "X", 1, 1, difficulty="hard")

        assert used == ["medium", "hard"]

    def test_random_move_rate_per_difficulty(self, monkeypatch):
        # random.choice picks the last empty square, (2, 2), so a random move is easy to recognise.
        game = TicTacToe()
        monkeypatch.setattr(tictactoe_webapp.random, "choice", lambda options: options[-1])

        def move_with_roll(roll, difficulty):
            monkeypatch.setattr(tictactoe_webapp.random, "random", lambda: roll)
            return tictactoe_webapp.choose_agent_move(StubAgent(), game, difficulty)

        random_move = (2, 2)
        learned_move = ("learned move", False)

        assert (
            move_with_roll(0.49, "easy") == random_move
            and move_with_roll(0.5, "easy") == learned_move
            and move_with_roll(0.19, "medium") == random_move
            and move_with_roll(0.2, "medium") == learned_move
            and move_with_roll(0.0, "hard") == learned_move
        )

    def test_make_move_ai_replies(self):
        board = [["X", "", ""], ["", "", ""], ["", "", ""]]
        response = post_move(board, "X", 1, 1)
        new_board = response.json()["board"]

        assert (
            response.status_code == 200
            and new_board[1][1] == "O"
            and sum(row.count("X") for row in new_board) == 2
        )

    def test_make_move_on_occupied_cell_is_invalid_move(self):
        board = [["X", "", ""], ["", "", ""], ["", "", ""]]
        response = post_move(board, "X", 0, 0)

        assert (
            response.status_code == 200
            and response.json()["message"] == "Invalid Move!"
            and response.json()["board"] == board
        )

    def test_make_move_rejects_wrong_piece_counts(self):
        # The human moved first, so X and O must have equal counts. Two O's and no X cannot happen.
        board = [["O", "O", ""], ["", "", ""], ["", "", ""]]
        response = post_move(board, "O", 0, 2)

        assert response.status_code == 400

    def test_make_move_rejects_finished_game(self):
        board = [["O", "O", "O"], ["X", "X", ""], ["X", "", ""]]
        response = post_move(board, "O", 2, 2)

        assert response.status_code == 400

    def test_make_move_rejects_unknown_cell_values(self):
        board = [["Z", "", ""], ["", "", ""], ["", "", ""]]
        response = post_move(board, "O", 1, 1)

        assert response.status_code == 400
