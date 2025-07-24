"""
Optimal strategy implementation using expected value calculations.
"""
from typing import List
from ...core.dice import DiceRoll
from ...core.scoring import ScoringEngine
from ...core.game_state import DecisionState
from .base import Strategy, StrategyConfig


class OptimalStrategy(Strategy):
    """
    Optimal strategy using expected value calculations.
    Stops when expected value of continuing is less than current score.
    """
    
    def setup(self, risk_adjustment: float = 1.0):
        self.risk_adjustment = risk_adjustment
        self._ev_cache = {}
        self._initialize_expected_values()
    
    def _initialize_expected_values(self):
        """Pre-calculate expected values for different dice counts."""
        # Expected values calculated through dynamic programming
        # These are approximations - in practice we'd calculate exact values
        self._base_ev = {
            1: 50,    # With 1 die: 1/3 chance of 100, 1/6 chance of 50
            2: 125,   # With 2 dice: higher chance of scoring
            3: 200,   # With 3 dice: possibility of triples
            4: 275,   # With 4 dice: good scoring chances
            5: 350,   # With 5 dice: very good chances
            6: 450,   # With 6 dice: excellent chances
        }
    
    def _calculate_continuation_value(self, game_state: DecisionState) -> float:
        """
        Calculate expected value of continuing from current state.
        This is a simplified version - a full implementation would use
        Monte Carlo or exact probability calculations.
        """
        remaining_dice = game_state.remaining_dice
        current_score = game_state.accumulated_score
        bonus = game_state.bonus_value
        achieved_tutto = game_state.achieved_tutto
        
        if remaining_dice == 0:
            # If no dice left, we'd roll all 6 again
            remaining_dice = 6
        
        # Special case for 1 die - exact calculation
        if remaining_dice == 1 and not achieved_tutto:
            # With 1 die: 1/6 chance of 1 (100 points + bonus), 1/6 chance of 5 (50 points + bonus), 4/6 bust
            ev_roll_1 = (1/6) * (current_score + 100 + bonus)  # Roll a 1, achieve tutto
            ev_roll_5 = (1/6) * (current_score + 50 + bonus)   # Roll a 5, achieve tutto
            ev_bust = (4/6) * 0  # Bust
            continuation_ev = ev_roll_1 + ev_roll_5 + ev_bust
            return continuation_ev
        
        # Base expected value for the number of dice
        base_ev = self._base_ev.get(remaining_dice, 100)
        
        # Adjust for current score (risk increases with higher scores)
        risk_factor = 1.0 - (current_score / 1000) * 0.2  # More conservative as score increases
        risk_factor = max(0.5, risk_factor)  # Don't go below 50% risk factor
        
        # Probability of busting (simplified)
        bust_prob = self._get_bust_probability(remaining_dice)
        
        # Expected value calculation
        # EV = P(success) * (current + expected_gain) + P(bust) * 0
        success_prob = 1 - bust_prob
        expected_gain = base_ev * risk_factor
        
        # If we can achieve tutto, add expected bonus value
        if not achieved_tutto and remaining_dice <= 3:
            tutto_prob = self._get_tutto_probability(remaining_dice)
            expected_gain += tutto_prob * bonus
        
        continuation_ev = success_prob * (current_score + expected_gain)
        
        # Apply risk adjustment parameter
        continuation_ev *= self.risk_adjustment
        
        return continuation_ev
    
    def _get_bust_probability(self, num_dice: int) -> float:
        """Get probability of busting with given number of dice."""
        # Approximate probabilities
        bust_probs = {
            1: 0.667,  # 4/6 chance of non-scoring
            2: 0.444,  # Approximately
            3: 0.296,
            4: 0.198,
            5: 0.132,
            6: 0.088,
        }
        return bust_probs.get(num_dice, 0.1)
    
    def _get_tutto_probability(self, num_dice: int) -> float:
        """Get probability of achieving tutto with remaining dice."""
        # Simplified probabilities
        tutto_probs = {
            1: 0.333,  # Must roll 1 or 5
            2: 0.25,   # Various combinations
            3: 0.15,   # Triple or all 1s/5s
            4: 0.08,
            5: 0.04,
            6: 0.02,
        }
        return tutto_probs.get(num_dice, 0.01)
    
    def should_continue(self, game_state: DecisionState) -> bool:
        """Continue if expected value of continuing exceeds current score."""
        if game_state.round_status != "in_progress":
            return False
        
        if game_state.remaining_dice == 0:
            # Must continue if achieved tutto
            return True
        
        current_value = game_state.total_score_if_stop
        continuation_value = self._calculate_continuation_value(game_state)
        
        return continuation_value > current_value
    
    def select_dice(self, roll: DiceRoll, scoring_engine: ScoringEngine) -> List[int]:
        """Select dice to maximize expected value."""
        score, combinations = scoring_engine.calculate_score(roll)
        
        # For now, select all scoring dice (optimal in most cases)
        # A more sophisticated approach would consider keeping only
        # high-value dice in certain situations
        selected_dice = []
        for combo in combinations:
            selected_dice.extend(combo.dice_used)
        
        return selected_dice
    
    @classmethod
    def get_default_config(cls) -> StrategyConfig:
        return StrategyConfig(
            name="Optimal",
            description="Stop when EV of continuing < current score",
            parameters={"risk_adjustment": 1.0}
        )