import random
from collections import Counter
from typing import List

from game_and_agent import QLearningAgent, TicTacToe, make_opponent_move
from time import perf_counter


def play_game_to_test_second_mover_agent(
    agent: QLearningAgent, agent1: QLearningAgent, episodes: int = 10000
) -> List[int]:
    """
    Play multiple games to test an agent who move second.

    Parameters:
    agent (QLearningAgent): A Q-learning agent object who move second.
    agent1 (QLearningAgent): A Q-learning agent object who move first.
    episodes (int): The number of episodes to play (default: 10000).
    """

    game_record = []
    average_reward = 0
    game = TicTacToe()

    # Play multiple games to train the Q-learning agent

    start = perf_counter()
    for episode in range(episodes):
        if episode > 0 and episode % 10000 == 0:
            print(
                f"Played {episode} episodes. AR is {average_reward}. Time used: {perf_counter() - start:.2f}"
            )

        # Reset the game board for a new game

        game.reset()

        # Player numbers on the board: 1 is the agent being tested, 2 is its opponent.
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
            action = agent.choose_action(state_key, valid_actions, is_learning=False)

            # Make the chosen move on the game board
            game.make_move(*action, agent_player)

            # Calculate the reward for the move. No need to check draw after the agent's move
            # because the 9th move is always made by opponent, if the opponent move first
            reward = -0.1
            if game.check_win(agent_player):
                reward = 1  # Winning the game
            else:
                # The opponent wins immediately if it can, otherwise agent1 chooses its move
                make_opponent_move(game, agent1, is_learning=True)

                # Calculate the reward for the move

                if game.check_win(opponent_player):
                    reward = -1
                elif game.check_draw():
                    reward = 0

            # average_reward is a exponential moving average of the reward when the game is in terminal states.
            if reward in [-1, 0, 1]:
                average_reward = 0.9999 * average_reward + (1 - 0.9999) * reward
                if reward == -1:
                    raise Exception("loss!")

                game_record.append(reward)
    return game_record


if __name__ == "__main__":
    # agent is the agent we are testing. Whenever it loses a game, we will issue an Exception. agent will move second.

    agent = QLearningAgent(
        pre_trained_q_table="q_table_ubuntu_agent_move_second.json",
    )

    # agent1 is the AI opponent. agent1 will always move first.

    agent1 = QLearningAgent(
        pre_trained_q_table="q_table_ubuntu_agent_move_first.json",
    )

    # Train the agent by playing the game
    EP = 50000
    game_record = play_game_to_test_second_mover_agent(agent, agent1, episodes=EP)
    if -1 not in set(game_record):
        counter = Counter(game_record)

        print(
            f"Among the {EP} games played, the second mover agent won {counter[1]} games and draw {counter[0]} games."
        )
