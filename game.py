import numpy as np

# victory_condition(game_state) => winner or None
# end_condition(game_state) => True (no more moves) or False
# player_strategy(game_state) => new game state


def run_turn_based_game(victory_condition, end_condition, player_strategies, initial_game_state,
                        print_fn=None):
    game_state = initial_game_state
    num_players = len(player_strategies)
    player = 0
    winner = None

    if print_fn:
        print_fn(game_state)

    while not winner and not end_condition(game_state):
        game_state = player_strategies[player](game_state)
        player = (player + 1) % num_players
        winner = victory_condition(game_state)

        if print_fn:
            print_fn(game_state)

    return winner, game_state