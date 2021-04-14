# The Gomoku Emulator

## Game

The Game class is the most important part of the emulator. Both of training and evaluation will interact with it. It holds the main loop and will callback the decision methods defined by training/evaluation models.
There are 2 methods related to training/evaluation:

`start_play(self, player1, player2, start_player=0, is_shown=1)`  
The method used by evaluation. 
- `player1`: the first player, which will determine what action will be taken during playing
- `player2`: the second player
- `start_player`: which player to be the first
- `is_shown`: whether to visualize the gomoku board to console

`start_self_play(self, player, is_shown=0, temp=1e-3)`  
The method used by training. 
- `player`: the self played player
- `is_shown`: whether to visualize the gomoku board to console
- `temp`: temperature parameter to control the level of exploration for AlphaZero


## Board

Board define the Gomoku board. It stores the game states, e.g. which position is already taken. The Gomoku logic is implemented in it as well, which will determine whether the game is end and whether there is a winner. **However it doesn't support the [“house” rules](https://en.wikipedia.org/wiki/Gomoku) of Gomoku**.

Training and evaluation won't have to interact with Board. It's width and height can be customized, as well as a `n_in_row` parameter to define how many stones in a row to win a game.

Several key methods are listed here:

`game_end(self)`  
Return whether the current game is end.

`has_a_winner(self)`  
Determine whether this game has a winner.

`do_move(self, move)`  
Do a move.

`current_state(self)`  
Return the internal state, i.e. each position's state
