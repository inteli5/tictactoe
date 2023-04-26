from typing import List

from game_and_agent import QLearningAgent, TicTacToe
from time import perf_counter
from collections import Counter


def play_game_to_test_first_mover_agent(
    agent: QLearningAgent, agent1: QLearningAgent, episodes: int = 10000
) -> List[int]:
    """
    Play multiple games to train the Q-learning agent.

    Parameters:
    agent (QLearningAgent): A Q-learning agent object who move first.
    agent1 (QLearningAgent): A Q-learning agent object who move second.
    episodes (int): The number of episodes to play (default: 10000).
    """

    game_record = []
    average_reward = 0
    game = TicTacToe()

    # Play multiple games to test the Q-learning agent

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

        state_key = game.get_state_key()

        # Continue playing until a player wins or the game is a draw
        while not (
            game.check_win(player1) or game.check_win(player2) or game.check_draw()
        ):
            valid_actions = game.get_valid_actions()

            # Choose an action based on the agent's exploration/exploitation strategy
            action = agent.choose_action(state_key, valid_actions, is_learning=False)

            # Make the chosen move on the game board
            game.make_move(*action, player1)

            # Calculate the reward for the move
            reward = -0.1
            if game.check_win(player1):
                reward = 1  # Winning the game
            elif game.check_draw():
                reward = 0  # Game is a draw
            else:
                # if the opponent can win in the next move, we will choose this to accelerate the learning process.
                use_check_win_move = False
                check_win_move_valid_actions = game.get_valid_actions()
                for check_win_action in check_win_move_valid_actions:
                    game.make_move(*check_win_action, player2)
                    if game.check_win(player2):
                        use_check_win_move = True
                        break
                    else:
                        game.withdraw_move(*check_win_action)
                if not use_check_win_move:
                    # ai opponent.
                    agent1_state_key = game.get_state_key().translate(
                        str.maketrans("12", "21")
                    )

                    agent1_action = agent1.choose_action(
                        agent1_state_key, game.get_valid_actions(), is_learning=True
                    )
                    game.make_move(*agent1_action, player2)

                # Calculate the reward for the move

                if game.check_win(player2):
                    reward = -1

            # Get the new state key after making the move
            state_key = game.get_state_key()

            if reward in [-1, 0, 1]:
                average_reward = (
                    0.9999 * average_reward + (1 - 0.9999) * reward
                )  # /(1-0.9999**(episode+1))
                if reward == -1:
                    raise Exception("loss!")

                game_record.append(reward)
    return game_record


if __name__ == "__main__":
    # agent is the agent we are testing. Whenever it loses a game, we will issue an Exception. agent will move first.
    agent = QLearningAgent(
        pre_trained_q_table="q_table_ubuntu_agent_move_first.pkl",
    )

    # agent1 is the AI opponent. agent1 will always move second.
    agent1 = QLearningAgent(
        pre_trained_q_table="q_table_ubuntu_agent_move_second.pkl",
    )

    # Train the agent by playing the game
    EP = 50000
    game_record = play_game_to_test_first_mover_agent(agent, agent1, episodes=EP)
    if -1 not in set(game_record):
        counter = Counter(game_record)

        print(
            f"Among the {EP} games played, the first mover agent won {counter[1]} games and draw {counter[0]} games."
        )
