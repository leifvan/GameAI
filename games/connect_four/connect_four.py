import numpy as np
from itertools import groupby
from operator import itemgetter
from random import choice
from functools import partial

from game import TurnBasedGame, run_multiple_games


class ConnectFourGame(TurnBasedGame):
    num_players = 2

    def get_initial_game_state(self):
        return np.zeros((6, 7), dtype=int)

    @staticmethod
    def victory_condition(game_state):
        zero_padded_state = np.pad(game_state, ((0, 1), (0, 1)), mode='constant')

        # check rows
        p, max_len = get_max_consecutive_row_length(zero_padded_state.ravel())
        if max_len == 4:
            return p

        # check cols
        p, max_len = get_max_consecutive_row_length(np.transpose(zero_padded_state).ravel())
        if max_len == 4:
            return p

        # check diagonals
        diagonals = np.concatenate([np.diagonal(zero_padded_state, offset=o) for o in range(-3, 4)])
        p, max_len = get_max_consecutive_row_length(diagonals)
        if max_len == 4:
            return p

        # check anti-diagonals
        diagonals = np.concatenate([np.diagonal(np.rot90(zero_padded_state), offset=o)
                                    for o in range(-4, 3)])
        p, max_len = get_max_consecutive_row_length(diagonals)
        if max_len == 4:
            return p

        return None

    @staticmethod
    def end_condition(game_state):
        return np.count_nonzero(game_state == 0) == 0

    @staticmethod
    def legal_state_condition(game_state):
        diff = np.diff((game_state != 0).astype(int), axis=0)
        return np.all(diff >= 0)


def get_max_consecutive_row_length(flat_rows):
    consecutive_row_lengths = ((k, sum(1 for _ in g)) for k, g in groupby(flat_rows)
                               if k != 0)
    try:
        return max(consecutive_row_lengths, key=itemgetter(1))
    except ValueError: # max() arg is an empty sequence
        return 0, 0


def get_legal_positions(game_state):
    one_padded_state = np.pad(game_state, ((0,1),(0,0)), mode='constant', constant_values=1)
    diff = np.diff((one_padded_state != 0).astype(int), axis=0)
    return np.transpose(np.nonzero(diff))


def random_strategy(game_state, p):
    positions = get_legal_positions(game_state)
    pos = choice(positions)
    game_state[pos[0], pos[1]] = p
    return game_state


def winning_or_random_move(game_state, p):
    positions = get_legal_positions(game_state)

    # check if victory is possible
    for pos in positions:
        game_state[pos[0], pos[1]] = p
        if ConnectFourGame.victory_condition(game_state):
            return game_state
        game_state[pos[0], pos[1]] = 0

    # check if victory of opponent has to be prevented
    op = 1 if p == 2 else 2
    for pos in positions:
        game_state[pos[0], pos[1]] = op
        if ConnectFourGame.victory_condition(game_state):
            game_state[pos[0], pos[1]] = p
            return game_state
        game_state[pos[0], pos[1]] = 0

    # pick random
    pos = choice(positions)
    game_state[pos[0], pos[1]] = p
    return game_state


if __name__ == '__main__':
    # game = ConnectFourGame()
    # game.game_state = np.array([[0,0,0,1,0,0,0],
    #                             [0,0,1,0,0,0,0],
    #                             [0,1,0,0,0,0,0],
    #                             [1,0,0,0,0,0,0],
    #                             [0,0,0,0,0,0,0],
    #                             [0,0,0,0,0,0,0],
    #                             [0,0,0,0,0,0,0]])
    # print(game.victory_condition(game.game_state))


    strats = [partial(winning_or_random_move, p=1),
              partial(random_strategy, p=2)]

    stats = run_multiple_games(ConnectFourGame, strats, num_games=1000)