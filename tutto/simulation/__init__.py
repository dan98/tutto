"""Simulation module for Tutto."""
from .game_simulator import GameSimulator, SimulationResult
from .strategies import Strategy
from .strategy_registry import strategy_registry

# Backward compatibility aliases
MonteCarloSimulator = GameSimulator
MultiPlayerSimulator = GameSimulator

__all__ = [
    "GameSimulator",
    "SimulationResult", 
    "Strategy",
    "strategy_registry",
    # Backward compatibility
    "MonteCarloSimulator",
    "MultiPlayerSimulator",
]