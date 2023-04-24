from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.responses import HTMLResponse
from game_and_agent import TicTacToe, QLearningAgent

app = FastAPI(title="Tic Tac Toe")
templates = Jinja2Templates(directory="templates")

game = TicTacToe()
agent1 = QLearningAgent(
    alpha=0.1, gamma=0.9, epsilon=0.1, pre_trained_q_table="q_table_ubuntu_agent_move_first.pkl"
)
agent2 = QLearningAgent(
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1,
    pre_trained_q_table="q_table_ubuntu_agent_move_second.pkl",
)
# game.reset()
# state_key = game.get_state_key()

player_agent = 1
player_human = 2


class GameState(BaseModel):
    board: List[List[str]]
    player_move_first: str
    message: str


class Item(BaseModel):
    state: GameState
    x: int
    y: int


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("tictactoe.html", {"request": request})


@app.post("/make_move")
# async def make_move(state: GameState, x: int, y: int):
async def make_move(item: Item):
    state = item.state
    state.message = ""

    x = item.x
    y = item.y
    if x not in range(3) or y not in range(3) or state.board[x][y] != "":
        state.message = "Invalid Move!"
        return state

    if state.player_move_first == "X":
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
    agent_x, agent_y = agent.choose_action(
        agent_state_key, game.get_valid_actions(), is_learning=False
    )

    state.board[agent_x][agent_y] = "X"
    game.make_move(agent_x, agent_y, player_agent)

    if game.check_win(player_agent):
        state.message = "AI wins!"
    if game.check_draw():
        state.message = "It is a draw!"

    return state


if __name__ == "__main__":
    uvicorn.run("tictactoe_web:app", host="192.168.86.57", port=8000, reload=True)
