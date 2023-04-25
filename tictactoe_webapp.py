from typing import List

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

# Initialize Tic Tac Toe game and QLearning agents
game = TicTacToe()
agent1 = QLearningAgent(
    pre_trained_q_table="q_table_ubuntu_agent_move_first.pkl",
)
agent2 = QLearningAgent(
    pre_trained_q_table="q_table_ubuntu_agent_move_second.pkl",
)

player_agent = 1
player_human = 2


class GameState(BaseModel):
    """
    Game state model representing the current state of the Tic Tac Toe game.
    """

    board: List[List[str]]
    player_who_move_first: str  # could be "X" or "O", "X" means AI.
    message: str


class Item(BaseModel):
    """
    Item model containing the game state and the next move.
    """

    state: GameState
    x: int
    y: int


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Route to serve the home page of the Tic Tac Toe web app.

    Args:
        request: The incoming request.

    Returns:
        The rendered template of the Tic Tac Toe game page.
    """
    return templates.TemplateResponse("tictactoe.html", {"request": request})


@app.post("/make_move")
async def make_move(item: Item) -> GameState:
    """
    Route to handle the player's move and the AI's response.

    Args:
        item: An item object containing the game state and the player's move.

    Returns:
        The updated game state after the player's move and the AI's response.
    """
    print(item)
    state = item.state
    state.message = ""

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
    board = np.array(state.board)
    board = np.where(board == "X", "1", board)
    board = np.where(board == "O", "2", board)
    board = np.where(board == "", "0", board)
    board = board.astype(int)

    game.set_board(board)

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
    uvicorn.run("tictactoe_webapp:app", host="0.0.0.0", port=8000, reload=True)
