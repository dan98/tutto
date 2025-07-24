"""
Probability threshold strategy implementation.
"""
from typing import List
from ...core.dice import DiceRoll
from ...core.scoring import ScoringEngine
from ...core.game_state import DecisionState
from .base import Strategy, StrategyConfig


class ProbabilityThresholdStrategy(Strategy):
    """Continue based on success probability threshold."""
    
    def setup(self, success_threshold: float = 0.6, min_score: int = 200):
        self.success_threshold = success_threshold
        self.min_score = min_score
    
    def should_continue(self, game_state: DecisionState) -> bool:
        if game_state.accumulated_score < self.min_score:
            return True  # Always continue if below minimum
        
        remaining_dice = game_state.remaining_dice
        
        if remaining_dice == 0:
            return True  # Must continue if tutto achieved
        
        # Use pre-calculated success probabilities
        success_probs = {
            1: 0.333,
            2: 0.556, 
            3: 0.704,
            4: 0.802,
            5: 0.868,
            6: 0.912,
        }
        
        success_prob = success_probs.get(remaining_dice, 0.5)
        return success_prob >= self.success_threshold
    
    def select_dice(self, roll: DiceRoll, scoring_engine: ScoringEngine) -> List[int]:
        score, combinations = scoring_engine.calculate_score(roll)
        selected_dice = []
        for combo in combinations:
            selected_dice.extend(combo.dice_used)
        return selected_dice
    
    @classmethod
    def get_default_config(cls) -> StrategyConfig:
        return StrategyConfig(
            name="ProbabilityThreshold",
            description="Continue if success probability exceeds threshold",
            parameters={"success_threshold": 0.6, "min_score": 200}
        )