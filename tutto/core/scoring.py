from typing import List, Tuple, Dict, Set
from collections import Counter
from dataclasses import dataclass
from .dice import DiceRoll


@dataclass
class ScoringCombination:
    """Represents a scoring combination."""
    dice_used: List[int]
    points: int
    description: str
    
    def __str__(self) -> str:
        return f"{self.description}: {self.points} points"


class ScoringEngine:
    """Handles all scoring logic for Tutto."""
    
    TRIPLE_SCORES = {
        1: 1000,
        2: 200,
        3: 300,
        4: 400,
        5: 500,
        6: 600
    }
    
    def calculate_score(self, roll: DiceRoll) -> Tuple[int, List[ScoringCombination]]:
        """Calculate total score and identify all scoring combinations."""
        combinations = []
        used_indices = set()
        value_counts = roll.value_counts
        
        # Check for triples first
        for value, count in value_counts.items():
            if count >= 3:
                triple_indices = [i for i, v in enumerate(roll.values) if v == value and i not in used_indices][:3]
                used_indices.update(triple_indices)
                combinations.append(ScoringCombination(
                    dice_used=[value] * 3,
                    points=self.TRIPLE_SCORES[value],
                    description=f"Triple {value}s"
                ))
        
        # Check for single 1s and 5s
        for i, value in enumerate(roll.values):
            if i not in used_indices:
                if value == 1:
                    combinations.append(ScoringCombination(
                        dice_used=[1],
                        points=100,
                        description="Single 1"
                    ))
                    used_indices.add(i)
                elif value == 5:
                    combinations.append(ScoringCombination(
                        dice_used=[5],
                        points=50,
                        description="Single 5"
                    ))
                    used_indices.add(i)
        
        total_score = sum(combo.points for combo in combinations)
        return total_score, combinations
    
    def get_all_valid_selections(self, roll: DiceRoll) -> List[Tuple[List[int], int]]:
        """Get all valid dice selections and their scores."""
        valid_selections = []
        
        # Get individual scoring dice
        scoring_indices = []
        for i, value in enumerate(roll.values):
            if value in [1, 5]:
                scoring_indices.append(i)
        
        # Check for triples
        value_counts = roll.value_counts
        triple_values = [v for v, count in value_counts.items() if count >= 3]
        
        # Generate all possible valid selections
        if not scoring_indices and not triple_values:
            return []  # No valid selections
        
        # For simplicity, return the best selection for now
        score, _ = self.calculate_score(roll)
        if score > 0:
            valid_selections.append((roll.values, score))
        
        return valid_selections
    
    def is_valid_selection(self, roll: DiceRoll, selected_dice: List[int]) -> bool:
        """Check if a selection of dice is valid (has at least one scoring die)."""
        selected_roll = DiceRoll(selected_dice)
        score, _ = self.calculate_score(selected_roll)
        return score > 0
    
    def has_scoring_dice(self, roll: DiceRoll) -> bool:
        """Check if the roll has any scoring dice."""
        score, _ = self.calculate_score(roll)
        return score > 0