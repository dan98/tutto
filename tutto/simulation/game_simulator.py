"""Game simulation engine for Tutto."""
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from ..core.game import Game, Round, RoundStatus
from ..core.player import Player, PlayerType
from ..core.cards import CardType, Card
from ..core.dice import DiceRoll
from ..core.game_state import DecisionState
from .strategies import Strategy
from .strategy_registry import strategy_registry


@dataclass
class SimulationResult:
    """Results from game simulations."""
    num_simulations: int
    win_rates: Dict[str, float]  # Player name -> win rate
    avg_scores: Dict[str, float]  # Player name -> average final score
    avg_rounds: float  # Average number of rounds per game
    score_distributions: Dict[str, np.ndarray]  # Player name -> score distribution
    
    def __str__(self) -> str:
        lines = [f"Simulation Results ({self.num_simulations} games):"]
        lines.append(f"Average game length: {self.avg_rounds:.1f} rounds")
        lines.append("\nWin Rates:")
        for player, rate in sorted(self.win_rates.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {player}: {rate:.1%}")
        lines.append("\nAverage Final Scores:")
        for player, score in sorted(self.avg_scores.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {player}: {score:.0f}")
        return "\n".join(lines)


class GameSimulator:
    """Simulates Tutto games with various player configurations."""
    
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.human_decision_callback: Optional[Callable] = None
    
    def set_human_decision_callback(self, callback: Callable):
        """Set callback function for human player decisions."""
        self.human_decision_callback = callback
    
    def simulate_games(
        self,
        player_configs: List[Tuple[str, PlayerType, Optional[str]]],
        num_simulations: int = 1000,
        target_score: int = 6000,
        human_callback: Optional[Callable] = None
    ) -> SimulationResult:
        """Simulate multiple games with given player configurations.
        
        Args:
            player_configs: List of (name, player_type, strategy_name) tuples
            num_simulations: Number of games to simulate
            target_score: Target score to win the game
            human_callback: Callback for human player decisions (for single game simulation)
        """
        # Check if we have human players
        has_human = any(pt == PlayerType.HUMAN for _, pt, _ in player_configs)
        
        if has_human and num_simulations > 1:
            raise ValueError("Cannot simulate multiple games with human players")
        
        if has_human:
            # Single game with human player
            self.human_decision_callback = human_callback
            result = self._play_single_game(player_configs, target_score)
            return self._create_result_from_single_game(result, player_configs)
        
        # Multiple AI games - use parallel processing
        simulations_per_worker = num_simulations // self.num_workers
        remaining = num_simulations % self.num_workers
        
        futures = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for i in range(self.num_workers):
                n_sims = simulations_per_worker + (1 if i < remaining else 0)
                if n_sims > 0:
                    future = executor.submit(
                        self._run_simulations,
                        player_configs,
                        n_sims,
                        target_score,
                        i  # Worker ID for different random seeds
                    )
                    futures.append(future)
            
            # Collect results
            all_results = []
            for future in as_completed(futures):
                results = future.result()
                all_results.extend(results)
        
        # Aggregate results
        return self._aggregate_results(all_results, player_configs)
    
    def _play_single_game(
        self,
        player_configs: List[Tuple[str, PlayerType, Optional[str]]],
        target_score: int
    ) -> Dict:
        """Play a single game, possibly with human players."""
        # Create players
        players = []
        strategies = {}
        
        for name, player_type, strategy_name in player_configs:
            player = Player(name, player_type)
            players.append(player)
            
            # Set up strategy for AI players
            if player_type != PlayerType.HUMAN and strategy_name:
                strategies[name] = strategy_registry.get_strategy(strategy_name)
        
        # Play game
        game = Game(players, target_score)
        rounds_played = 0
        
        while not game.winner:
            rounds_played += 1
            if rounds_played > 1000:  # Safety limit
                break
            
            current_player = game.current_player
            round_obj, card = game.start_new_round()
            
            if not round_obj:  # Stop card
                continue
            
            # Play the round
            if current_player.player_type == PlayerType.HUMAN:
                self._play_human_round(round_obj, game)
            else:
                self._play_round_with_strategy(
                    round_obj, 
                    strategies.get(current_player.name),
                    game
                )
            
            game.complete_round()
        
        # Return game result
        return {
            "winner": game.winner.name if game.winner else None,
            "rounds": rounds_played,
            "final_scores": {p.name: p.score for p in players},
        }
    
    @staticmethod
    def _run_simulations(
        player_configs: List[Tuple[str, PlayerType, Optional[str]]],
        num_simulations: int,
        target_score: int,
        worker_id: int
    ) -> List[Dict]:
        """Run simulations in a single process."""
        np.random.seed(worker_id)  # Different seed per worker
        results = []
        
        for _ in range(num_simulations):
            # Create players and strategies
            players = []
            strategies = {}
            
            for name, player_type, strategy_name in player_configs:
                player = Player(name, player_type)
                players.append(player)
                
                # Create strategy
                if strategy_name:
                    strategies[name] = strategy_registry.get_strategy(strategy_name)
                elif player_type == PlayerType.AI_OPTIMAL:
                    strategies[name] = strategy_registry.get_strategy("optimal")
                elif player_type == PlayerType.AI_GREEDY:
                    strategies[name] = strategy_registry.get_strategy("greedy")
                elif player_type == PlayerType.AI_CONSERVATIVE:
                    strategies[name] = strategy_registry.get_strategy("conservative")
            
            # Play game
            game = Game(players, target_score)
            rounds_played = 0
            
            while not game.winner:
                rounds_played += 1
                if rounds_played > 1000:  # Safety limit
                    break
                
                current_player = game.current_player
                round_obj, card = game.start_new_round()
                
                if not round_obj:  # Stop card
                    continue
                
                # Play the round using strategy
                GameSimulator._play_round_with_strategy(
                    round_obj, 
                    strategies[current_player.name],
                    game
                )
                
                game.complete_round()
            
            # Record game result
            result = {
                "winner": game.winner.name if game.winner else None,
                "rounds": rounds_played,
                "final_scores": {p.name: p.score for p in players},
            }
            results.append(result)
        
        return results
    
    def _play_human_round(self, round_obj: Round, game: Game):
        """Handle a round for a human player."""
        if not self.human_decision_callback:
            raise ValueError("No human decision callback provided")
        
        card = round_obj.state.card
        
        while round_obj.state.status == RoundStatus.IN_PROGRESS:
            # Get current state
            decision_state = round_obj.get_decision_state(game)
            
            # Check if player must continue
            if self._must_continue(decision_state):
                # Player has no choice
                should_continue = True
            else:
                # Ask human for decision
                should_continue = self.human_decision_callback(
                    "continue", 
                    decision_state,
                    None  # No additional data
                )
            
            if not should_continue:
                round_obj.stop_round()
                break
            
            # Roll dice
            roll, score, _ = round_obj.roll_dice()
            if round_obj.state.status == RoundStatus.BUSTED:
                break
            
            # Get dice selection
            valid_selections = self._get_valid_dice_selections(round_obj, roll)
            
            if len(valid_selections) == 1:
                # Only one valid selection
                selected_dice = valid_selections[0]
            else:
                # Ask human to select
                selected_dice = self.human_decision_callback(
                    "select_dice",
                    decision_state,
                    {"roll": roll, "valid_selections": valid_selections}
                )
                
                # Validate selection
                if selected_dice not in valid_selections:
                    raise ValueError(f"Invalid dice selection: {selected_dice}")
            
            round_obj.select_dice(selected_dice)
    
    @staticmethod
    def _play_round_with_strategy(round_obj: Round, strategy: Strategy, game: Game):
        """Play a round using an AI strategy."""
        card = round_obj.state.card
        
        while round_obj.state.status == RoundStatus.IN_PROGRESS:
            # Get decision state with game context
            decision_state = round_obj.get_decision_state(game)
            
            # Check if must continue
            if GameSimulator._must_continue(decision_state):
                should_continue = True
            else:
                should_continue = strategy.should_continue(decision_state)
            
            if not should_continue:
                round_obj.stop_round()
                break
            
            # Roll dice
            roll, score, _ = round_obj.roll_dice()
            if round_obj.state.status == RoundStatus.BUSTED:
                break
            
            # Select dice based on card rules
            valid_selections = GameSimulator._get_valid_dice_selections(round_obj, roll)
            
            if len(valid_selections) == 1:
                # Only one valid selection (e.g., Fireworks must take all)
                selected_dice = valid_selections[0]
            else:
                # Use strategy to select
                selected_dice = strategy.select_dice(roll, round_obj.scoring_engine)
                
                # Ensure selection is valid
                if selected_dice not in valid_selections:
                    # Find best valid selection
                    selected_dice = GameSimulator._find_best_valid_selection(
                        valid_selections, 
                        round_obj.scoring_engine
                    )
            
            if selected_dice:
                round_obj.select_dice(selected_dice)
    
    @staticmethod
    def _must_continue(decision_state: DecisionState) -> bool:
        """Check if player must continue based on card rules."""
        card_type = decision_state.card_type
        
        if card_type == CardType.FIREWORKS:
            # Must continue until bust
            return True
        elif card_type == CardType.PLUS_MINUS:
            # Must continue until tutto
            return not decision_state.achieved_tutto
        elif card_type == CardType.CLOVERLEAF:
            # Must continue until 2 tuttos
            return decision_state.tutto_count < 2
        elif card_type == CardType.STRAIGHT:
            # Must continue until complete
            return not decision_state.straight_complete
        
        return False
    
    @staticmethod
    def _get_valid_dice_selections(round_obj: Round, roll: DiceRoll) -> List[List[int]]:
        """Get all valid dice selections based on card rules."""
        card = round_obj.state.card
        
        if card.card_type == CardType.FIREWORKS:
            # Must keep ALL scoring dice
            all_scoring = GameSimulator._get_all_scoring_dice(roll, round_obj.scoring_engine)
            return [all_scoring] if all_scoring else []
        
        elif card.card_type == CardType.STRAIGHT:
            # Must keep one die per new unique value
            new_values = set(roll.values) - round_obj.state.straight_collected
            if not new_values:
                return []
            
            # Create selection with one die per new value
            selected = []
            for value in sorted(new_values):
                for v in roll.values:
                    if v == value and v not in selected:
                        selected.append(v)
                        break
            return [selected]
        
        else:
            # Normal selection - any valid scoring combination
            return GameSimulator._get_all_valid_selections(roll, round_obj.scoring_engine)
    
    @staticmethod
    def _get_all_scoring_dice(roll: DiceRoll, scoring_engine) -> List[int]:
        """Get all dice that contribute to score."""
        scoring_dice = []
        used_indices = set()
        
        # Get all combinations
        score, combinations = scoring_engine.calculate_score(roll)
        
        # First, handle triples
        from collections import Counter
        counts = Counter(roll.values)
        for val, count in counts.items():
            if count >= 3:
                # Find indices of this triple
                indices = [i for i, v in enumerate(roll.values) if v == val]
                for idx in indices[:3]:  # Take first 3
                    scoring_dice.append(roll.values[idx])
                    used_indices.add(idx)
        
        # Then add remaining 1s and 5s
        for i, val in enumerate(roll.values):
            if i not in used_indices and (val == 1 or val == 5):
                scoring_dice.append(val)
        
        return scoring_dice
    
    @staticmethod
    def _get_all_valid_selections(roll: DiceRoll, scoring_engine) -> List[List[int]]:
        """Get all possible valid dice selections."""
        score, combinations = scoring_engine.calculate_score(roll)
        
        if score == 0:
            return []
        
        valid_selections = []
        
        # Add each individual combination
        for combo in combinations:
            valid_selections.append(combo.dice_used)
        
        # Also add the option of taking all scoring dice
        all_scoring = GameSimulator._get_all_scoring_dice(roll, scoring_engine)
        if all_scoring and all_scoring not in valid_selections:
            valid_selections.append(all_scoring)
        
        return valid_selections
    
    @staticmethod
    def _find_best_valid_selection(valid_selections: List[List[int]], scoring_engine) -> List[int]:
        """Find the best selection from valid options."""
        if not valid_selections:
            return []
        
        # For now, return the selection with highest immediate score
        best_score = 0
        best_selection = valid_selections[0]
        
        for selection in valid_selections:
            roll = DiceRoll(selection)
            score, _ = scoring_engine.calculate_score(roll)
            if score > best_score:
                best_score = score
                best_selection = selection
        
        return best_selection
    
    @staticmethod
    def _aggregate_results(
        all_results: List[Dict],
        player_configs: List[Tuple[str, PlayerType, Optional[str]]]
    ) -> SimulationResult:
        """Aggregate results from all simulations."""
        num_games = len(all_results)
        player_names = [config[0] for config in player_configs]
        
        # Initialize counters
        wins = {name: 0 for name in player_names}
        total_scores = {name: 0 for name in player_names}
        score_lists = {name: [] for name in player_names}
        total_rounds = 0
        
        # Process results
        for result in all_results:
            if result["winner"]:
                wins[result["winner"]] += 1
            total_rounds += result["rounds"]
            
            for name, score in result["final_scores"].items():
                total_scores[name] += score
                score_lists[name].append(score)
        
        # Calculate statistics
        win_rates = {name: wins[name] / num_games for name in player_names}
        avg_scores = {name: total_scores[name] / num_games for name in player_names}
        avg_rounds = total_rounds / num_games
        score_distributions = {name: np.array(scores) for name, scores in score_lists.items()}
        
        return SimulationResult(
            num_simulations=num_games,
            win_rates=win_rates,
            avg_scores=avg_scores,
            avg_rounds=avg_rounds,
            score_distributions=score_distributions
        )
    
    def _create_result_from_single_game(
        self, 
        game_result: Dict,
        player_configs: List[Tuple[str, PlayerType, Optional[str]]]
    ) -> SimulationResult:
        """Create a SimulationResult from a single game."""
        player_names = [config[0] for config in player_configs]
        
        # Create win rates (1 for winner, 0 for others)
        win_rates = {name: 0.0 for name in player_names}
        if game_result["winner"]:
            win_rates[game_result["winner"]] = 1.0
        
        # Scores are just the final scores
        avg_scores = game_result["final_scores"]
        score_distributions = {name: np.array([score]) for name, score in avg_scores.items()}
        
        return SimulationResult(
            num_simulations=1,
            win_rates=win_rates,
            avg_scores=avg_scores,
            avg_rounds=float(game_result["rounds"]),
            score_distributions=score_distributions
        )