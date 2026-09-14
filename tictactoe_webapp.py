from typing import List, Literal

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.responses import HTMLResponse
from game_and_agent import TicTacToe, QLearningAgent

# Create FastAPI app and Jinja2 templates
app = FastAPI(title="Tic Tac Toe")
templates = Jinja2Templates(directory="templates")

# Initialize QLearning agents.
# Every request builds its own TicTacToe game from the board it receives, so requests never share a board.
agent1 = QLearningAgent(
    pre_trained_q_table="q_table_ubuntu_agent_move_first.json",
)
agent2 = QLearningAgent(
    pre_trained_q_table="q_table_ubuntu_agent_move_second.json",
)

player_agent = 1
player_human = 2


class GameState(BaseModel):
    """
    Game state model representing the current state of the Tic Tac Toe game.
    """

    board: List[List[str]]
    player_who_move_first: Literal["X", "O"]  # "X" means AI.
    message: str


class Item(BaseModel):
    """
    Item model containing the game state and the next move.
    """

    state: GameState
    x: int
    y: int


class NewGame(BaseModel):
    """
    Request model for starting a new game.
    """

    player_who_move_first: Literal["X", "O"]  # "X" means AI.


def board_to_game(board: List[List[str]]) -> TicTacToe:
    """
    Create a TicTacToe game from the board sent by the page.

    Args:
        board: A 3x3 list with "X" (AI), "O" (human) or "" (empty) in each cell.

    Returns:
        A TicTacToe game whose board uses 1 for the AI, 2 for the human and 0 for empty cells.
    """
    symbol_to_player = {"X": player_agent, "O": player_human, "": 0}
    game = TicTacToe()
    game.set_board(
        np.array([[symbol_to_player[cell] for cell in row] for row in board], dtype=int)
    )
    return game


def validate_board(state: GameState) -> None:
    """
    Reject a board that cannot occur in a real game, so a forged request cannot, for example, claim a win.

    The board is checked before the human's move is added:
    - it must be 3x3, and every cell must be "X", "O" or "";
    - if the AI (X) moved first, there is one more X than O; if the human (O) moved first, the counts are equal;
    - nobody has won yet.

    Args:
        state: The game state sent by the page.

    Raises:
        HTTPException: 400 if the board cannot occur in a real game.
    """
    board = state.board
    if len(board) != 3 or any(len(row) != 3 for row in board):
        raise HTTPException(status_code=400, detail="The board must be 3x3.")
    if any(cell not in ("X", "O", "") for row in board for cell in row):
        raise HTTPException(
            status_code=400, detail='Every cell must be "X", "O" or "".'
        )

    x_count = sum(row.count("X") for row in board)
    o_count = sum(row.count("O") for row in board)
    expected_x_count = o_count + 1 if state.player_who_move_first == "X" else o_count
    if x_count != expected_x_count:
        raise HTTPException(
            status_code=400,
            detail="The numbers of X and O do not match who moved first.",
        )

    game = board_to_game(board)
    if game.check_win(player_agent) or game.check_win(player_human):
        raise HTTPException(status_code=400, detail="The game is already over.")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Route to serve the home page of the Tic Tac Toe web app.

    Args:
        request: The incoming request.

    Returns:
        The rendered template of the Tic Tac Toe game page.
    """
    return templates.TemplateResponse(request, "tictactoe.html")


@app.post("/new_game")
async def new_game(new_game_request: NewGame) -> GameState:
    """
    Route to start a new game.

    If the AI moves first, it makes its opening move here with its learned Q-table (which picks a corner),
    instead of the page placing the first X at random.

    Args:
        new_game_request: Who moves first, "X" (AI) or "O" (human).

    Returns:
        The state of the new game, including the AI's opening move if the AI moves first.
    """
    state = GameState(
        board=[["", "", ""], ["", "", ""], ["", "", ""]],
        player_who_move_first=new_game_request.player_who_move_first,
        message="",
    )

    if state.player_who_move_first == "X":
        game = TicTacToe()
        x, y = agent1.choose_action(
            game.get_state_key(), game.get_valid_actions(), is_learning=False
        )
        state.board[x][y] = "X"

    return state


@app.post("/make_move")
async def make_move(item: Item) -> GameState:
    """
    Route to handle the player's move and the AI's response.

    Args:
        item: An item object containing the game state and the player's move.

    Returns:
        The updated game state after the player's move and the AI's response.

    Raises:
        HTTPException: 400 if the board sent by the page cannot occur in a real game.
    """
    state = item.state
    state.message = ""
    validate_board(state)

    x = item.x
    y = item.y
    if x not in range(3) or y not in range(3) or state.board[x][y] != "":
        state.message = "Invalid Move!"
        return state

    if state.player_who_move_first == "X":
        agent = agent1
    else:
        agent = agent2
    state.board[x][y] = "O"
    game = board_to_game(state.board)

    if game.check_win(player_human):
        state.message = "You win!"
        return state
    if game.check_draw():
        state.message = "It is a draw!"
        return state

    agent_state_key: str = game.get_state_key()

    # If you want the AI to lose sometimes, is_learning can be set to True.
    # Otherwise, AI will not lose.
    x, y = agent.choose_action(
        agent_state_key, game.get_valid_actions(), is_learning=False
    )

    state.board[x][y] = "X"
    game.make_move(x, y, player_agent)

    if game.check_win(player_agent):
        state.message = "AI wins!"
    if game.check_draw():
        state.message = "It is a draw!"

    return state


if __name__ == "__main__":
    # Only reachable from this computer.
    # For auto-reload during development, run: uv run uvicorn tictactoe_webapp:app --reload
    uvicorn.run(app, host="127.0.0.1", port=8000)
