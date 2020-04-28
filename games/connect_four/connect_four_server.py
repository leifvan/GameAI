from bottle import route, run, static_file, get, patch, request, abort
from games.connect_four.connect_four import ConnectFourGame, winning_or_random_move


def game_state_to_list(arr):
    return [[int(v) for v in row] for row in arr]


def game_to_dict(game):
    return {'gameState': game_state_to_list(game.game_state),
            'winner': int(game.winner) if game.winner else None,
            'player': game.player+1}


if __name__ == '__main__':
    game = ConnectFourGame()


    @route('/')
    def index_page():
        return static_file("game.html", root=".")


    @get("/gamestate")
    def get_game_state():
        return game_to_dict(game)


    @patch("/restart")
    def restart_game():
        global game
        game = ConnectFourGame()
        return game_to_dict(game)


    @patch("/move")
    def make_move():
        circle_id = request.query.id
        i, j = map(int, circle_id)

        if game.game_ended:
            abort(400, text="Game ended.")

        if game.game_state[i, j] != 0:
            abort(400, text="Illegal move.")

        game.game_state[i, j] = game.player + 1
        if not game.legal_state_condition(game.game_state):
            game.game_state[i, j] = 0
            abort(400, text="Illegal move.")

        game.move(game.game_state)

        if not game.winner and not game.game_ended:
            game.move(winning_or_random_move(game.game_state, game.player+1))

        return game_to_dict(game)

    run(host='localhost', port=8000)
