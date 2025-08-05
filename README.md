# ConnectK

A ConnectK (Connect Four variant) AI bot implementation using minimax algorithm with strategic analysis.

## Project Overview

ConnectK is a generalized version of Connect Four where players try to connect K pieces in a row on an M×N board. This project contains AI bots that can play ConnectK tournaments.

## Features

The main bot includes:

- Minimax algorithm with alpha-beta pruning (3-depth search)
- Threat detection for different types of winning patterns
- Center column preference and strategic positioning
- Win detection and opponent blocking

Key capabilities:

1. Immediate win detection
2. Defensive blocking
3. Threat analysis and prevention
4. Strategic center control
5. Position evaluation

## Project Structure

```
ConnectK/
├── connectk_engine.exe     # Tournament execution engine
├── LICENSE                 # MIT License
├── README.md              # Project documentation
└── bots/
    ├── bot.py             # Main AI bot with strategic gameplay
    └── user_bot.py        # User customizable bot
```

## Getting Started

### Requirements

- Python 3.11+ (required by the tournament engine)
- Windows OS (the tournament engine is a Windows executable)

### Running a Tournament

1. Clone this repository:

   ```bash
   git clone https://github.com/tansexe/ConnectK.git
   cd ConnectK
   ```

2. Run the tournament:

   ```cmd
   connectk_engine.exe
   ```

3. Follow the prompts to set up the game (board size, connect length, bot selection)

## How the AI Works

The bot uses a minimax algorithm to evaluate moves several steps ahead. It looks for:

- Immediate winning moves
- Moves that block the opponent from winning
- Threats that force the opponent to respond
- Good strategic positions (especially center control)

The bot evaluates positions based on:

- Potential winning lines
- Control of center columns
- Piece connectivity
- Threat patterns

## Customizing the Bot

You can modify the bot's behavior by editing `bot.py`:

- Change the search depth in `get_best_move_lookahead()`
- Adjust scoring weights for different strategies
- Modify center column preferences
- Add your own evaluation criteria

To create your own bot, use `user_bot.py` as a starting point:

```python
def next_move(board):
    # Your bot logic here
    # Return column index (0-based)
    pass
```

## File Descriptions

- **bot.py**: The main AI bot with minimax search and threat analysis
- **user_bot.py**: A template for creating custom bots
- **connectk_engine.exe**: Tournament runner that handles game logic and scoring

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

Taniya Nawal Pathak - [tansexe](https://github.com/tansexe)
