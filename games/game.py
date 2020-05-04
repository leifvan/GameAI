from abc import ABC, abstractmethod
from collections import defaultdict
from collections import deque
from typing import Type

from tqdm import tqdm


# victory_condition(game_state) => winner or None
# end_condition(game_state) => True (no more moves) or False
# player_strategy(game_state) => new game state


class GameEndedException(Exception):
    pass


class IllegalMoveException(Exception):
    pass


class TurnBasedGame(ABC):
    num_players = 1

    def __init__(self):
        self.game_state = self.get_initial_game_state()
        self.player = 0
        self.winner = None
        self.game_ended = False

    @staticmethod
    @abstractmethod
    def get_initial_game_state():
        ...

    @staticmethod
    @abstractmethod
    def victory_condition(game_state):
        ...

    @staticmethod
    @abstractmethod
    def end_condition(game_state):
        ...

    @staticmethod
    @abstractmethod
    def legal_state_condition(game_state):
        ...

    @staticmethod
    @abstractmethod
    def possible_moves(game_state):
        ...

    @staticmethod
    @abstractmethod
    def make_move(game_state, move):
        ...

    def print_state(self):
        print(self.game_state)

    def move(self, new_game_state):
        if self.game_ended:
            raise GameEndedException

        if not self.legal_state_condition(new_game_state):
            raise IllegalMoveException

        self.game_state = new_game_state
        self.player = (self.player + 1) % self.num_players

        self.winner = self.victory_condition(self.game_state)
        self.game_ended = self.winner or self.end_condition(self.game_state)

    def run(self, player_strategies, print_states=False):
        if print_states:
            self.print_state()

        while not self.winner and not self.game_ended:
            self.move(player_strategies[self.player](self.game_state))

            if print_states:
                self.print_state()

        return self.winner, self.game_state

    @classmethod
    def run_multiple_games(cls, player_strategies, num_games):
        wins = defaultdict(int)

        with tqdm(range(num_games)) as pbar:
            for i in pbar:
                winner, _ = cls().run(player_strategies)
                wins[winner] += 1
                if i % 10 == 0:
                    pbar.set_postfix({f'p{p}' if p else 'draw': w for p, w in wins.items()})

        return wins

    @classmethod
    def get_full_game_tree(cls, initial_state=None, starting_player=0):
        initial_state = initial_state or cls.get_initial_game_state()
        game_tree = GameTree(cls, initial_state, starting_player)

        with tqdm() as pbar:
            expand_queue = deque([(game_tree.root, initial_state)])

            while len(expand_queue) > 0:
                node, state = expand_queue.popleft()
                game_tree.expand_node(node, state)
                expand_queue.extend([(child, cls.make_move(state, child.move)) for child in node.children])
                pbar.update(1)

        return game_tree


class GameTreeNode:
    def __init__(self, move, parent):
        self.move = move
        self.parent = parent
        self.children = []

    def path_to_root(self):
        path = [self]
        while path[-1].parent:
            path.append(path[-1].parent)
        return path


class GameTree:
    def __init__(self, game_type: Type[TurnBasedGame], initial_state=None, starting_player=0):
        self.game_type = game_type

        if initial_state is None:
            self.initial_state = game_type.get_initial_game_state()
        else:
            self.initial_state = initial_state

        self.starting_player = starting_player
        self.root = GameTreeNode(move=None, parent=None)

    def get_state(self, node: GameTreeNode):
        path = node.path_to_root()[::-1]
        state = self.initial_state

        for pnode in path[1:]:
            state = self.game_type.make_move(state, pnode.move)

        return state

    def get_statistics(self):
        num_nodes = num_leafs = num_branches = 0
        stack = [self.root]

        while len(stack) > 0:
            node = stack.pop()
            num_nodes += 1

            if len(node.children) == 0:
                num_leafs += 1
            else:
                num_branches += len(node.children)

            stack.extend(node.children)

        return num_nodes, num_leafs, num_branches

    def expand_node(self, node: GameTreeNode, state=None):
        if state is None:
            state = self.get_state(node)

        if self.game_type.victory_condition(state) or self.game_type.end_condition(state):
            return

        possible_moves = self.game_type.possible_moves(state)

        for move in possible_moves:
            node.children.append(GameTreeNode(move=move, parent=node))
