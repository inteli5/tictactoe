from fastapi.testclient import TestClient

from tictactoe_webapp import app

client = TestClient(app)

CORNERS = [(0, 0), (0, 2), (2, 0), (2, 2)]


def post_move(board, player_who_move_first, x, y):
    state = {"board": board, "player_who_move_first": player_who_move_first, "message": ""}
    return client.post("/make_move", json={"state": state, "x": x, "y": y})


class TestWebApp:
    def test_home_page_loads(self):
        response = client.get("/")

        assert response.status_code == 200 and '<div class="grid">' in response.text

    def test_new_game_ai_first_opens_with_learned_corner_move(self):
        response = client.post("/new_game", json={"player_who_move_first": "X"})
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
