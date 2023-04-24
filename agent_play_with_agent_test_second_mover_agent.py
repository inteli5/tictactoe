import random
from collections import Counter
from typing import List

from game_and_agent import QLearningAgent, TicTacToe
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

        # Set the first player (1 for 'X' and 2 for 'O'), 'X' is agent, 'O' is opponent
        player1 = 1
        player2 = 2

        # The opponent will move first in the game.
        # Get the valid actions for the current state for player2
        player2_valid_actions = game.get_valid_actions()

        # Choose an action for player2
        player2_random_actions = random.choice(player2_valid_actions)

        # Make the chosen move on the game board
        game.make_move(*player2_random_actions, player2)

        # Continue playing until a player wins or the game is a draw
        while not (
            game.check_win(player1) or game.check_win(player2) or game.check_draw()
        ):
            # Get the current state key
            state_key = game.get_state_key()

            valid_actions = game.get_valid_actions()

            # Choose an action based on the agent's exploration/exploitation strategy
            action = agent.choose_action(state_key, valid_actions, is_learning=False)

            # Make the chosen move on the game board
            game.make_move(*action, player1)

            # Calculate the reward for the move. No need to check draw after the agent's move
            # because the 9th move is always made by opponent, if the opponent move first
            reward = -0.1
            if game.check_win(player1):
                reward = 1  # Winning the game
            else:
                # if the opponent can win in the next move, we will choose this to accelerate the learning process.
                use_check_win_move = False
                valid_actions = game.get_valid_actions()
                for i in valid_actions:
                    game.make_move(*i, player2)
                    if game.check_win(player2):
                        use_check_win_move = True
                        break
                    else:
                        game.withdraw_move(*i)
                if not use_check_win_move:
                    # ai opponent.
                    agent1_state_key = game.get_state_key().translate(
                        str.maketrans("12", "21")
                    )

                    agent1_action = agent1.choose_action(
                        agent1_state_key, game.get_valid_actions()
                    )
                    game.make_move(*agent1_action, player2)

                # Calculate the reward for the move

                if game.check_win(player2):
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
        alpha=0.1,
        gamma=1,
        epsilon=0.1,
        pre_trained_q_table="q_table_ubuntu_agent_move_second.pkl",
    )

    # agent1 is the AI opponent. agent1 will always move first.

    agent1 = QLearningAgent(
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        pre_trained_q_table="q_table_ubuntu_agent_move_first.pkl",
    )

    # Train the agent by playing the game
    EP = 50000
    game_record = play_game_to_test_second_mover_agent(agent, agent1, episodes=EP)
    if -1 not in set(game_record):
        counter = Counter(game_record)

        print(
            f"Among the {EP} games played, the second mover agent won {counter[1]} games and draw {counter[0]} games."
        )
