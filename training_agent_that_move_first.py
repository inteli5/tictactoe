from time import perf_counter

from game_and_agent import QLearningAgent, TicTacToe, make_opponent_move


def play_game_agent_move_first(agent: QLearningAgent, episodes: int = 10000) -> None:
    """
    Play multiple games to train the Q-learning agent with a random opponent.

    Parameters:
    agent (QLearningAgent): A Q-learning agent object who move first. The agent to be trained.
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
        # The agent moves first.
        agent_player = 1
        opponent_player = 2

        state_key = game.get_state_key()

        # Continue playing until a player wins or the game is a draw
        while not (
            game.check_win(agent_player) or game.check_win(opponent_player) or game.check_draw()
        ):
            # Get the current state key
            valid_actions = game.get_valid_actions()

            # Choose an action based on the agent's exploration/exploitation strategy
            action = agent.choose_action(state_key, valid_actions)

            # Make the chosen move on the game board
            game.make_move(*action, agent_player)

            # Calculate the reward for the move
            reward = -0.1
            if game.check_win(agent_player):
                reward = 1  # Winning the game
            elif game.check_draw():
                reward = 0  # Game is a draw
            else:
                # The opponent wins immediately if it can, otherwise it plays a random move
                make_opponent_move(game)

                # Calculate the reward for the move

                if game.check_win(opponent_player):
                    reward = -1

            # Get the new state key after making the move
            next_state_key = game.get_state_key()

            # Get the valid actions for the next state
            # If we are in the terminal state, the next_valid_actions will be [], so the next_max_q_value is 0.
            # We update the Q value by Q(s, a) ← Q(s, a) + α[r - Q(s, a)]
            next_valid_actions = (
                game.get_valid_actions() if reward not in [-1, 0, 1] else []
            )

            # Update the Q-table using the Q-learning update rule

            agent.learn(state_key, action, reward, next_state_key, next_valid_actions)

            # Update the state_key for the next iteration
            state_key = next_state_key

            if reward in [-1, 0, 1]:
                average_reward = (
                    0.9999 * average_reward + (1 - 0.9999) * reward
                )  # /(1-0.9999**(episode+1))


if __name__ == "__main__":
    agent = QLearningAgent()

    # Train the agent by playing the game
    EP = 1000000
    play_game_agent_move_first(agent, episodes=EP)

    agent.save_q_table("q_table_ubuntu_agent_move_first.json")
