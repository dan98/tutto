"""
Registry for managing and accessing different strategies.
"""
from typing import Dict, Type, List, Optional
from .strategies import (
    Strategy, StrategyConfig, GreedyStrategy, ConservativeStrategy,
    OptimalStrategy, ProbabilityThresholdStrategy
)
from .game_theoretic_strategy import (
    GameTheoreticOptimalStrategy, AdaptiveStrategy
)


class StrategyRegistry:
    """Registry for managing available strategies."""
    
    def __init__(self):
        self._strategies: Dict[str, Type[Strategy]] = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register all built-in strategies."""
        self.register(GreedyStrategy)
        self.register(ConservativeStrategy)
        self.register(OptimalStrategy)
        self.register(ProbabilityThresholdStrategy)
        self.register(GameTheoreticOptimalStrategy)
        self.register(AdaptiveStrategy)
    
    def register(self, strategy_class: Type[Strategy]):
        """Register a new strategy class."""
        config = strategy_class.get_default_config()
        self._strategies[config.name.lower()] = strategy_class
    
    def get_strategy(self, name: str, **kwargs) -> Strategy:
        """Get a strategy instance by name with optional parameter overrides."""
        strategy_class = self._strategies.get(name.lower())
        if not strategy_class:
            raise ValueError(f"Unknown strategy: {name}")
        
        # Get default config and update with provided kwargs
        config = strategy_class.get_default_config()
        if kwargs:
            config.parameters.update(kwargs)
        
        return strategy_class(config)
    
    def list_strategies(self) -> List[str]:
        """List all available strategy names."""
        return list(self._strategies.keys())
    
    def get_strategy_info(self, name: str) -> StrategyConfig:
        """Get information about a strategy."""
        strategy_class = self._strategies.get(name.lower())
        if not strategy_class:
            raise ValueError(f"Unknown strategy: {name}")
        return strategy_class.get_default_config()
    
    def get_all_strategies_info(self) -> Dict[str, StrategyConfig]:
        """Get information about all registered strategies."""
        return {
            name: strategy_class.get_default_config()
            for name, strategy_class in self._strategies.items()
        }


# Global registry instance
strategy_registry = StrategyRegistry()