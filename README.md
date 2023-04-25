# Tic Tac Toe with FastAPI and Reinforcement learning

A project of the Tic Tac Toe game with FastAPI and Reinforcement learning.

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

After training your own agent, you can test it by running
```bash
python agent_play_with_agent_test_first_mover_agent.py
```
or 
```bash
python agent_play_with_agent_test_second_mover_agent.py
```
If you set the parameters well, your agents should never lose. 

The file 'game_and_agent.py' is the game board and reinforcement learning agent class. It accelerates the learning by updating not only the current state-action but also the symmetrical state-action pairs.

## License

[MIT](https://choosealicense.com/licenses/mit/)