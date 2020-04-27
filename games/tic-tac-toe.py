import numpy as np
from functools import partial
from collections import defaultdict
from tqdm import tqdm

from games.game import run_turn_based_game

rng = np.random.default_rng()


def victory_condition(game_state):
    if move_was_winning_move(game_state, 1):
        return 0
    elif move_was_winning_move(game_state, -1):
        return 1
    return None


def end_condition(game_state):
    return not move_still_possible(game_state)


def move_still_possible(S):
    return not (S[S == 0].size == 0)


def move_at_random(S, p):
    xs, ys = np.where(S == 0)
    i = rng.integers(0, xs.size)
    S[xs[i], ys[i]] = p
    return S


def move_at_random_weighted(game_state, p, weights):
    positions = np.flatnonzero(game_state == 0)

    relevant_weights = np.ravel(weights)[positions]
    relevant_weights /= np.sum(relevant_weights)

    target = rng.choice(positions, p=relevant_weights)
    target = np.unravel_index(target, game_state.shape)
    game_state[target] = p
    return game_state


def select_winning_move_or_weighted(game_state, p, weights):
    positions = np.transpose(np.nonzero(game_state == 0))

    # check if other player can win and block
    for px, py in positions:
        game_state[px,py] = -p
        if move_was_winning_move(game_state, -p):
            game_state[px,py] = p
            return game_state
        game_state[px,py] = 0

    # check if this player can win
    for px, py in positions:
        game_state[px,py] = p
        if move_was_winning_move(game_state, p):
            return game_state
        game_state[px,py] = 0

    # choose randomly
    return move_at_random_weighted(game_state, p, weights)


def move_was_winning_move(S, p):
    if np.max((np.sum(S, axis=0)) * p) == 3:
        return True

    if np.max((np.sum(S, axis=1)) * p) == 3:
        return True

    if (np.sum(np.diag(S)) * p) == 3:
        return True

    if (np.sum(np.diag(np.rot90(S))) * p) == 3:
        return True

    return False


# python dictionary to map integers (1, -1, 0) to characters ('x', 'o', ' ')
symbols = {1: 'x',
           -1: 'o',
           0: ' '}


# print game state matrix using characters
def print_game_state(game_state):
    for row in game_state:
        print(' '.join(symbols[v] for v in row))
    print("-"*5)


if __name__ == '__main__':

    # completely random strategies
    random_strats = [partial(move_at_random, p=p) for p in [1, -1]]

    # player 1 chooses randomly with distribution from auspicious positions after 100,000 random games
    win_probs = np.array([[0.11811497, 0.09795438, 0.11796465],
                          [0.09668166, 0.14078844, 0.09746834],
                          [0.11683974, 0.09710506, 0.11708276]])

    auspicious_strats = [partial(move_at_random_weighted, p=1, weights=win_probs),
                         partial(move_at_random, p=-1)]

    # player 1 checks for a winning move before using auspicious position strategy
    winning_move_strats = [partial(select_winning_move_or_weighted, p=1, weights=win_probs),
                           partial(move_at_random, p=-1)]


    strats_to_use = winning_move_strats

    # --------------------------------------
    # run a single game and print all states
    # --------------------------------------

    winner, _ = run_turn_based_game(victory_condition=victory_condition,
                                    end_condition=end_condition,
                                    player_strategies=strats_to_use,
                                    initial_game_state=np.zeros((3,3), dtype=int),
                                    print_fn=print_game_state)

    print("player", winner, "won the game")

    # ------------------------------
    # run 10,000 games and evaluate:
    # ------------------------------

    wins = defaultdict(int)
    auspicious_positions = np.zeros((3, 3), dtype=int)

    for i in tqdm(range(10000)):
        winner, final_state = run_turn_based_game(victory_condition=victory_condition,
                                                  end_condition=end_condition,
                                                  player_strategies=strats_to_use,
                                                  initial_game_state=np.zeros((3, 3), dtype=int))
        wins[winner] += 1

        if winner == 0:
            auspicious_positions[final_state == 1] += 1
        elif winner == 1:
            auspicious_positions[final_state == -1] += 1

    print("x:   ", wins[0])
    print("o:   ", wins[1])
    print("draw:", wins[None])

    print("auspicious positions:")
    print(auspicious_positions / np.sum(auspicious_positions))
