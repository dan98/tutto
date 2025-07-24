import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from ..core.game import Round, RoundStatus
from ..core.dice import DiceRoll
from ..core.scoring import ScoringEngine
from ..core.cards import CardType
from ..core.player import Player, PlayerType
from .strategies import Strategy
from .strategy_registry import strategy_registry


@dataclass
class SimulationResult:
    """Results from a Monte Carlo simulation."""
    num_simulations: int
    success_rate: float
    expected_value: float
    variance: float
    std_deviation: float
    percentiles: Dict[int, float]
    score_distribution: np.ndarray
    busted_count: int
    tutto_count: int
    
    def __str__(self) -> str:
        return (
            f"Simulation Results ({self.num_simulations} runs):\n"
            f"  Success Rate: {self.success_rate:.1%}\n"
            f"  Expected Value: {self.expected_value:.1f}\n"
            f"  Std Deviation: {self.std_deviation:.1f}\n"
            f"  25th Percentile: {self.percentiles[25]:.1f}\n"
            f"  50th Percentile: {self.percentiles[50]:.1f}\n"
            f"  75th Percentile: {self.percentiles[75]:.1f}\n"
            f"  Bust Rate: {self.busted_count/self.num_simulations:.1%}\n"
            f"  Tutto Rate: {self.tutto_count/self.num_simulations:.1%}"
        )




class MonteCarloSimulator:
    """Runs Monte Carlo simulations for decision making."""
    
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers or multiprocessing.cpu_count()
    
    def simulate_continuation(
        self,
        current_state: dict,
        num_simulations: int = 10000,
        strategy: Optional[Union[Strategy, str]] = None,
        force_continue: bool = True
    ) -> SimulationResult:
        """Simulate continuing from current game state.
        
        Args:
            current_state: Current game state
            num_simulations: Number of simulations to run
            strategy: Strategy to use for decisions after the initial continuation
            force_continue: If True, forces at least one roll even if strategy says stop
        """
        if strategy is None:
            strategy = strategy_registry.get_strategy("greedy")
        elif isinstance(strategy, str):
            strategy = strategy_registry.get_strategy(strategy)
        
        # Split simulations across workers
        simulations_per_worker = num_simulations // self.num_workers
        remaining = num_simulations % self.num_workers
        
        futures = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for i in range(self.num_workers):
                n_sims = simulations_per_worker + (1 if i < remaining else 0)
                if n_sims > 0:
                    future = executor.submit(
                        self._run_simulations,
                        current_state,
                        n_sims,
                        strategy,
                        force_continue
                    )
                    futures.append(future)
            
            # Collect results
            all_scores = []
            total_busted = 0
            total_tutto = 0
            
            for future in as_completed(futures):
                scores, busted, tutto = future.result()
                all_scores.extend(scores)
                total_busted += busted
                total_tutto += tutto
        
        # Calculate statistics
        scores_array = np.array(all_scores)
        success_rate = np.sum(scores_array > current_state["accumulated_score"]) / len(scores_array)
        
        return SimulationResult(
            num_simulations=num_simulations,
            success_rate=success_rate,
            expected_value=np.mean(scores_array),
            variance=np.var(scores_array),
            std_deviation=np.std(scores_array),
            percentiles={
                25: np.percentile(scores_array, 25),
                50: np.percentile(scores_array, 50),
                75: np.percentile(scores_array, 75),
                95: np.percentile(scores_array, 95),
            },
            score_distribution=scores_array,
            busted_count=total_busted,
            tutto_count=total_tutto,
        )
    
    @staticmethod
    def _run_simulations(
        initial_state: dict,
        num_simulations: int,
        strategy: Strategy,
        force_continue: bool = True
    ) -> Tuple[List[float], int, int]:
        """Run simulations in a single process."""
        scores = []
        busted_count = 0
        tutto_count = 0
        
        for _ in range(num_simulations):
            # Get card from state or create a default one
            if "card" in initial_state:
                card = initial_state["card"]
            else:
                # For backward compatibility
                card_type = initial_state.get("card_type", CardType.BONUS)
                if card_type == CardType.STRAIGHT:
                    from ..core.cards import Card
                    card = Card(CardType.STRAIGHT)
                else:
                    from ..core.cards import Card
                    bonus_value = initial_state.get("bonus_card", 300)
                    card = Card(CardType.BONUS, bonus_value)
            
            # Create a dummy player for the round
            from ..core.player import Player
            dummy_player = Player("Simulator", PlayerType.AI_OPTIMAL)
            
            # Create a new round from the current state
            round = Round(card, dummy_player)
            round.state.accumulated_score = initial_state["accumulated_score"]
            round.state.achieved_tutto = initial_state.get("achieved_tutto", False)
            round.dice.remaining_dice = initial_state["remaining_dice"]
            
            # For Straight, set collected values
            if card.card_type == CardType.STRAIGHT:
                round.state.straight_collected = initial_state.get("straight_collected", initial_state.get("strasse_collected", set())).copy()
                round.state.straight_complete = initial_state.get("straight_complete", initial_state.get("strasse_complete", False))
            
            if card.card_type == CardType.STRAIGHT:
                # For Straight, no strategy decisions - just roll until complete or bust
                while round.state.status == RoundStatus.IN_PROGRESS and not round.state.straight_complete:
                    roll, score, combinations = round.roll_dice()
                    
                    if round.state.status == RoundStatus.BUSTED:
                        busted_count += 1
                        break
                    
                    # For Straight, select one die per new value
                    new_values = set(roll.values) - round.state.straight_collected
                    if new_values:
                        selected = []
                        for value in sorted(new_values):
                            # Find first occurrence of this value
                            for v in roll.values:
                                if v == value and v not in selected:
                                    selected.append(v)
                                    break
                        round.select_dice(selected)
                    
                    if round.state.straight_complete:
                        tutto_count += 1  # Count as "tutto" equivalent
            else:
                # Regular BONUS card logic
                # Force at least one roll if force_continue is True
                first_roll = True
                
                # Continue playing with strategy
                while (first_roll and force_continue) or (strategy.should_continue(round.get_decision_info()) and round.state.status == RoundStatus.IN_PROGRESS):
                    first_roll = False
                    roll, score, combinations = round.roll_dice()
                    
                    if round.state.status == RoundStatus.BUSTED:
                        busted_count += 1
                        break
                    
                    # Select dice using strategy
                    selected_dice = strategy.select_dice(roll, round.scoring_engine)
                    if selected_dice:
                        round.select_dice(selected_dice)
                        
                        if round.state.achieved_tutto:
                            tutto_count += 1
            
            # Stop if not busted
            if round.state.status == RoundStatus.IN_PROGRESS:
                round.stop_round()
            
            scores.append(round.state.total_score)
        
        return scores, busted_count, tutto_count
    
    def calculate_probabilities(self, num_dice: int) -> Dict[str, float]:
        """Calculate probabilities for different outcomes with given number of dice."""
        # Simulate many rolls to estimate probabilities
        num_trials = 100000
        scoring_count = 0
        tutto_possible = 0
        
        for _ in range(num_trials):
            values = np.random.randint(1, 7, size=num_dice)
            roll = DiceRoll(list(values))
            
            scoring_engine = ScoringEngine()
            score, _ = scoring_engine.calculate_score(roll)
            
            if score > 0:
                scoring_count += 1
                # Check if all dice score (tutto possible)
                if all(v in [1, 5] or values.tolist().count(v) >= 3 for v in set(values)):
                    tutto_possible += 1
        
        return {
            "scoring_probability": scoring_count / num_trials,
            "bust_probability": 1 - (scoring_count / num_trials),
            "tutto_possibility": tutto_possible / num_trials,
        }