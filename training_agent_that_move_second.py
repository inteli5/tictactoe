import random
from time import perf_counter

from game_and_agent import QLearningAgent, TicTacToe, make_opponent_move


def play_game_agent_move_second(
    agent: QLearningAgent, agent1: QLearningAgent, episodes: int = 10000
) -> None:
    """
    Play multiple games to train the Q-learning agent with a pre-trained AI opponent.

    Parameters:
    agent (QLearningAgent): A Q-learning agent object who move second.The agent to be trained.
    agent1 (QLearningAgent): A Q-learning agent object who move first. A pre-trained agent.
    episodes (int): The number of episodes to play (default: 10000).
    """

    average_reward = 0
    game = TicTacToe()

    # Play multiple games to train the Q-learning agent

    start = perf_counter()
    for episode in range(episodes):
        if episode > 0 and episode % 10000 == 0:
            agent.alpha *= 0.99

        if episode > 0 and episode % 10000 == 0:
            print(
                f"Trained {episode} episodes. LR is {agent.alpha}. AR is {average_reward}. Time used: {perf_counter() - start:.2f}"
            )

        # Reset the game board for a new game

        game.reset()

        # Player numbers on the board: 1 is the agent being trained, 2 is its opponent.
        agent_player = 1
        opponent_player = 2

        # The opponent will move first in the game.
        # Get the valid actions for the current state for the opponent
        opponent_valid_actions = game.get_valid_actions()

        # Choose an action for the opponent
        opponent_random_action = random.choice(opponent_valid_actions)

        # Make the chosen move on the game board
        game.make_move(*opponent_random_action, opponent_player)

        # Continue playing until a player wins or the game is a draw
        while not (
            game.check_win(agent_player) or game.check_win(opponent_player) or game.check_draw()
        ):
            # Get the current state key
            state_key = game.get_state_key()

            valid_actions = game.get_valid_actions()

            # Choose an action based on the agent's exploration/exploitation strategy
            # if episode<=EP:
            action = agent.choose_action(state_key, valid_actions)
            # else:
            #     action = agent.choose_action(state_key, valid_actions, is_learning=False)

            # Make the chosen move on the game board
            game.make_move(*action, agent_player)

            # Calculate the reward for the move. No need to check draw after the agent's move
            # because the 9th move is always made by opponent, if the opponent move first
            reward = -0.1
            if game.check_win(agent_player):
                reward = 1  # Winning the game
            else:
                # The opponent wins immediately if it can, otherwise agent1 chooses its move
                make_opponent_move(game, agent1)

                # Calculate the reward for the move

                if game.check_win(opponent_player):
                    reward = -1
                elif game.check_draw():
                    reward = 0

            # Get the new state key after making the move
            next_state_key = game.get_state_key()

            # Get the valid actions for the next state
            # If we are in the terminal state, the next_valid_actions will be [], so the next_max_q_value is 0.
            # We update the Q value by Q(s, a) ← Q(s, a) + α[r - Q(s, a)]
            next_valid_actions = (
                game.get_valid_actions() if reward not in [-1, 0, 1] else []
            )

            # average_reward is an exponential moving average of the reward when the game is in terminal states.
            if reward in [-1, 0, 1]:
                average_reward = 0.9999 * average_reward + 0.0001 * reward

            # Update the Q-table using the Q-learning update rule

            agent.learn(state_key, action, reward, next_state_key, next_valid_actions)


if __name__ == "__main__":
    # The agent to be trained
    agent = QLearningAgent()

    # agent1 is the AI opponent. agent1 will always move first. q_table_ubuntu_agent_move_first.json is a pre-trained
    # q_table for the first mover agent. The opponent agent can not always take
    # the optimal move, because if so, the agent can not learn from the opponent's suboptimal moves and could make wrong
    # move in the unseen states.
    agent1 = QLearningAgent(
        epsilon=0.5,
        pre_trained_q_table="q_table_ubuntu_agent_move_first.json",
    )

    # Train the agent by playing the game
    EP = 1000000
    play_game_agent_move_second(agent, agent1, episodes=EP)

    agent.save_q_table("q_table_ubuntu_agent_move_second.json")
