import numpy as np
from itertools import groupby
from operator import itemgetter
from random import choice
from functools import partial
from numba import njit
from collections import defaultdict
from tqdm import tqdm

from game import TurnBasedGame, GameTree


class ConnectFourGame(TurnBasedGame):
    num_players = 2

    @staticmethod
    def get_initial_game_state():
        return np.zeros((6, 7), dtype=int)

    @staticmethod
    def victory_condition(game_state, zero_padded_state=np.zeros((8, 9), dtype=np.int)):
        # zero_padded_state = np.pad(game_state, ((0, 0), (1, 1)), mode='constant')
        zero_padded_state[1:-1, 1:-1] = game_state

        # check rows
        p, max_len = get_max_consecutive_row_length(zero_padded_state.ravel())
        if max_len == 4:
            return p - 1

        # check cols
        p, max_len = get_max_consecutive_row_length(np.transpose(zero_padded_state).ravel())
        if max_len == 4:
            return p - 1

        # check diagonals
        diagonals = concat_diagonals(zero_padded_state)
        p, max_len = get_max_consecutive_row_length(diagonals)
        if max_len == 4:
            return p - 1

        # check anti-diagonals
        diagonals = concat_diagonals(np.fliplr(zero_padded_state))
        p, max_len = get_max_consecutive_row_length(diagonals)
        if max_len == 4:
            return p - 1

        return None

    @staticmethod
    def end_condition(game_state):
        return np.count_nonzero(game_state == 0) == 0

    @staticmethod
    def legal_state_condition(game_state):
        diff = np.diff((game_state != 0).astype(int), axis=0)
        return np.all(diff >= 0)

    @staticmethod
    def possible_moves(game_state):
        return get_legal_positions(game_state)

    @staticmethod
    def make_move(game_state, move):
        player = np.count_nonzero(game_state) % 2
        new_state = game_state.copy()
        assert new_state[move[0], move[1]] == 0
        new_state[move[0], move[1]] = player + 1
        return new_state

    def print_state(self):
        for row in self.game_state:
            print(*row)
        print("-"*(2*self.game_state.shape[1]-1))


@njit
def concat_diagonals(arr):
    max_len = min(arr.shape)
    num_items = 0

    # count
    for o in range(-3, 5):
        for i in range(max_len):
            if 0 <= i + o < arr.shape[1]:
                num_items += 1

    diags = np.zeros(num_items, dtype=np.int32)

    c = 0
    for o in range(-3, 5):
        for i in range(max_len):
            if 0 <= i + o < arr.shape[1]:
                diags[c] = arr[i, i + o]
                c += 1

    return diags


@njit
def get_max_consecutive_row_length(flat_rows):
    max_len = 0
    max_x = 0

    cur_len = 0
    cur_x = -1

    for x in flat_rows:
        if x != cur_x:
            if cur_x != 0 and max_len < cur_len:
                max_len = cur_len
                max_x = cur_x
            cur_len = 0
            cur_x = x
        cur_len += 1

    return max_x, max_len


def get_legal_positions(game_state):
    one_padded_state = np.pad(game_state, ((0, 1), (0, 0)), mode='constant', constant_values=1)
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


class SelectBestMinMaxStrategy:
    def __init__(self, player_type, tie_strategy, game_tree, depth=2 ** 32, eval_fn=None, tree=None):
        self.game_tree = game_tree
        self.cur_node = self.game_tree.root
        self.player_type = player_type
        self.tie_strategy = tie_strategy
        self.depth = depth
        self.eval_fn = eval_fn
        self.tree = tree

    def __call__(self, game_state):
        # reset if game_state is initial
        if np.count_nonzero(game_state) < 2:
            self.cur_node = self.game_tree.root

        # find the node for corresponding game state
        for child in self.cur_node.children:
            if np.allclose(self.game_tree.get_state(child), game_state):
                self.cur_node = child
                break

        _, best_node = self.cur_node.mmv(self.player_type, self.tie_strategy, self.depth, self.eval_fn, self.tree)
        new_state = ConnectFourGame.make_move(game_state, best_node.move)
        self.cur_node = best_node
        return new_state


zero_padded_state = np.zeros((8, 9), dtype=np.int)


def eval_for_player(game_state, player):
    mask = game_state == player
    zero_padded_state[1:-1, 1:-1] = mask
    value = 0
    value += get_max_consecutive_row_length(zero_padded_state.ravel())[1] ** 2

    value += get_max_consecutive_row_length(np.transpose(zero_padded_state).ravel())[1] ** 2

    diagonals = concat_diagonals(zero_padded_state)
    value += get_max_consecutive_row_length(diagonals)[1] ** 2

    diagonals = concat_diagonals(np.fliplr(zero_padded_state))
    value += get_max_consecutive_row_length(diagonals)[1] ** 2

    return value


def eval_fn(node, game_tree):
    game_state = game_tree.get_state(node)
    zero_padded_state[1:-1, 1:-1] = game_state
    # check rows
    value = eval_for_player(game_state, 1) - eval_for_player(game_state, 2)
    print(game_state)
    print(value)
    return value


if __name__ == '__main__':
    # game = ConnectFourGame()
    # game.game_state = np.array([[0,0,0,0,0,0,0],
    #                             [0,0,0,0,0,0,0],
    #                             [0,0,0,0,0,0,0],
    #                             [2,0,0,0,0,0,1],
    #                             [0,2,0,0,0,1,0],
    #                             [0,0,2,0,1,0,0],
    #                             [0,0,0,2,0,0,0]])
    # print(game.victory_condition(game.game_state))
    #
    # exit(0)

    strats = [partial(winning_or_random_move, p=1),
              partial(winning_or_random_move, p=2)]

    game_tree = GameTree(ConnectFourGame, terminal_utilities=[1000,-1000])

    strats = [SelectBestMinMaxStrategy('max', 'first', game_tree, depth=1, eval_fn=partial(eval_fn, game_tree=game_tree),
                                       tree=game_tree),
              SelectBestMinMaxStrategy('min', 'first', game_tree, depth=1, eval_fn=partial(eval_fn, game_tree=game_tree),
                                       tree=game_tree)]

    winner, _ = ConnectFourGame().run(strats, print_states=True)
    print(winner, "wins")

    wins = defaultdict(int)
    exit(0)

    with tqdm(range(100)) as pbar:
        for _ in pbar:
            winner, _ = ConnectFourGame().run(strats)
            wins[winner] += 1
            pbar.set_postfix_str(str(wins))
    # stats = run_multiple_games(ConnectFourGame, strats, num_games=1000)

    print(wins)
