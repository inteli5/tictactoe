# Tic Tac Toe with FastAPI and Reinforcement learning

A project of the Tic Tac Toe game with FastAPI web app and Reinforcement learning . The code features learning by updating symmetrical state-action q-value.

## Installation

```bash
git clone https://github.com/inteli5/tictactoe.git
```
create a virtual environment by, for example, 

```bash
conda create -n tictactoe python=3.10

```

```bash
conda activate tictactoe
```


Install required packages.
```bash
python -m pip install -r requirements.txt
```


## Usage

### Web App

In the root directory of the project,
```bash
python tictactoe_webapp.py
```
Then you can open a web browser and input the URL 127.0.0.1:8000
You are always 'O'. AI is always 'X'.
By default, AI move first. You can click the button "You (O) first" to move first.

### Training

q_table_ubuntu_agent_move_first.pkl is the q-table for the agent who move first.
q_table_ubuntu_agent_move_second.pkl is the q-table for the agent who move second.

You can train your own first mover agent by backing up the pkl files and running
```bash
python training_agent_that_move_first.py
```
This program trains a first mover agent by playing with a random opponent. No pkl file needed.

Train the second mover agent by running
```bash
python training_agent_that_move_second.py
```
This program requires the pkl file 'q_table_ubuntu_agent_move_first.pkl'.
It trains a second mover agent by playing with an AI opponent who use the above q-table.

After training your own agent, you can test it by running the following codes that let the AI agent plays with another AI agent.
To test the first mover agent, you can run,
```bash
python agent_play_with_agent_test_first_mover_agent.py
```
For testing second mover agent, you can run,
```bash
python agent_play_with_agent_test_second_mover_agent.py
```
If you set the parameters well, your agents should never lose. 
In the above test, the opponent agent (the agent not being tested) will not always follow the optimal move. It has 10% of change to take a random move.
If both agent follow the optimal moves, they will always draw.
You can try this by modify the following code. For the 'agent_play_with_agent_test_first_mover_agent.py', line 78
```python
agent1_action = agent1.choose_action(
    agent1_state_key, game.get_valid_actions(), is_learning=False)
```
Change the is_learning flag to False. 
For agent_play_with_agent_test_second_mover_agent.py, you can make similar changes.

The file 'game_and_agent.py' is the game board and reinforcement learning agent class. It accelerates the learning by updating not only the current state-action but also the symmetrical state-action pairs.
For example, here is part of the second mover agent q-table.

{('000020000', (0, 1)): -0.6744205096465703,

 ('000020000', (1, 0)): -0.6744205096465703,

 ('000020000', (2, 1)): -0.6744205096465703,

 ('000020000', (1, 2)): -0.6744205096465703,

 ('000020000', (0, 0)): -0.09080339631767263,

 ('000020000', (2, 0)): -0.09080339631767263,

 ('000020000', (2, 2)): -0.09080339631767263,

 ('000020000', (0, 2)): -0.09080339631767263}

'000020000' is the board state. It means that the opponent (always is 2) move first and place in the center of the board. Tuple like (0, 1) is the action. You can see that the first four actions are the four edges. They share the same q-value. The next four actions are the four corners. 
If the AI agent chooses the four edges, it will be sure to lose. So they share a lower q-value.
if the AI agent chooses the four corners, it can be a draw.

## License

[MIT](https://choosealicense.com/licenses/mit/)