from collections import defaultdict
from abc import ABC, abstractmethod
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


def run_multiple_games(game_type, player_strategies, num_games):
    wins = defaultdict(int)

    with tqdm(range(num_games)) as pbar:
        for i in pbar:
            winner, _ = game_type().run(player_strategies)
            wins[winner] += 1
            if i % 10 == 0:
                pbar.set_postfix({f'p{p}' if p else 'draw': w for p,w in wins.items()})

    return wins