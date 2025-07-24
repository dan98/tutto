# Tutto Dice Game Simulator

A Python implementation of the Tutto dice game with Monte Carlo simulation for optimal decision making.

## Features

- **Interactive CLI**: Play the game with a rich terminal interface
- **Monte Carlo Simulation**: Analyze probabilities and expected values for decision support
- **Scoring System**: Implements full Tutto scoring rules
- **Statistical Analysis**: Get detailed statistics on game outcomes

## Installation

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Interactive Mode

```bash
python main.py
```

Or after installation:

```bash
tutto
```

### Game Rules

- 6 standard dice
- Draw a bonus card (100-600 points)
- Roll all dice and collect scoring combinations:
  - Single 1: 100 points
  - Single 5: 50 points
  - Triple 1s: 1000 points
  - Triple 2s: 200 points
  - Triple 3s: 300 points
  - Triple 4s: 400 points
  - Triple 5s: 500 points
  - Triple 6s: 600 points
- Must keep at least one scoring die per roll
- If no scoring dice: BUST (lose all points)
- If all dice score: TUTTO (get bonus card value added)

### Simulation Features

The simulator provides:
- Expected value calculations
- Success/bust probability
- Score distribution analysis
- Optimal strategy recommendations

## Project Structure

```
tutto/
├── core/           # Game logic
│   ├── dice.py     # Dice and rolling mechanics
│   ├── scoring.py  # Scoring rules
│   └── game.py     # Game and round management
├── simulation/     # Monte Carlo simulation
│   └── simulator.py
├── cli/           # Command-line interface
│   └── interface.py
└── utils/         # Utility functions
```