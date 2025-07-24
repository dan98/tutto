"""
Test module for comparing different strategies in multi-player games.
"""
import numpy as np
from typing import Dict, List, Tuple
from tutto.simulation import GameSimulator
from tutto.core.player import PlayerType
from tutto.simulation.strategy_registry import strategy_registry
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.progress import track


@dataclass
class StrategyTestResult:
    """Results from strategy comparison tests."""
    strategy_name: str
    win_rate: float
    avg_score: float
    avg_winning_score: float
    avg_rounds_to_win: float
    games_won: int
    total_games: int


class TestStrategyComparison:
    """Test different strategies against each other in multi-player games."""
    
    def get_simulator(self):
        """Create a game simulator instance."""
        return GameSimulator(num_workers=4)
    
    def test_three_player_strategy_comparison(self, simulator=None):
        if simulator is None:
            simulator = self.get_simulator()
        """Test 3 players with different strategies competing."""
        # Define player configurations
        player_configs = [
            ("Optimal Player", PlayerType.AI_OPTIMAL, "optimal"),
            ("Conservative Player", PlayerType.AI_CONSERVATIVE, "conservative"),
            ("Greedy Player", PlayerType.AI_GREEDY, "greedy"),
        ]
        
        # Run simulations
        num_simulations = 1000
        print(f"\nRunning {num_simulations} games with 3 players using different strategies...")
        
        result = simulator.simulate_games(
            player_configs=player_configs,
            num_simulations=num_simulations,
            target_score=6000
        )
        
        # Analyze results
        self._print_results(result, player_configs)
        
        # Assert that results are reasonable
        # Verify results
        total_win_rate = sum(result.win_rates.values())
        assert abs(total_win_rate - 1.0) < 0.01, f"Total win rate {total_win_rate} not close to 1.0"
        assert all(0 <= rate <= 1 for rate in result.win_rates.values()), "Invalid win rates"
        assert all(score > 0 for score in result.avg_scores.values()), "Invalid scores"
    
    def test_all_strategies_comparison(self, simulator=None):
        if simulator is None:
            simulator = self.get_simulator()
        """Test all available strategies against each other."""
        # Get all available strategies
        all_strategies = strategy_registry.get_all_strategies_info()
        strategy_names = list(all_strategies.keys())
        
        # Create player configurations for each strategy
        player_configs = [
            (f"{name.capitalize()} Player", PlayerType.AI_OPTIMAL, name)
            for name in strategy_names[:4]  # Limit to 4 players for reasonable game length
        ]
        
        # Run simulations
        num_simulations = 500
        print(f"\nRunning {num_simulations} games with {len(player_configs)} players using all strategies...")
        
        result = simulator.simulate_games(
            player_configs=player_configs,
            num_simulations=num_simulations,
            target_score=6000
        )
        
        # Analyze results
        self._print_results(result, player_configs)
        
        # Create detailed strategy results
        strategy_results = self._create_strategy_results(result, player_configs, num_simulations)
        self._print_detailed_strategy_analysis(strategy_results)
    
    def test_optimal_vs_probability_threshold(self, simulator=None):
        if simulator is None:
            simulator = self.get_simulator()
        """Detailed comparison between optimal and probability threshold strategies."""
        player_configs = [
            ("Optimal 1", PlayerType.AI_OPTIMAL, "optimal"),
            ("Optimal 2", PlayerType.AI_OPTIMAL, "optimal"),
            ("Probability", PlayerType.AI_CONSERVATIVE, "probabilitythreshold"),
        ]
        
        num_simulations = 1000
        print(f"\nRunning {num_simulations} games: 2 Optimal vs 1 Probability Threshold...")
        
        result = simulator.simulate_games(
            player_configs=player_configs,
            num_simulations=num_simulations,
            target_score=6000
        )
        
        self._print_results(result, player_configs)
        
        # Check if optimal strategy dominates
        optimal_total_wins = result.win_rates.get("Optimal 1", 0) + result.win_rates.get("Optimal 2", 0)
        prob_wins = result.win_rates.get("Probability", 0)
        
        print(f"\nOptimal strategy total win rate: {optimal_total_wins:.1%}")
        print(f"Probability threshold win rate: {prob_wins:.1%}")
    
    def test_strategy_performance_by_player_count(self, simulator=None):
        if simulator is None:
            simulator = self.get_simulator()
        """Test how strategies perform with different numbers of players."""
        strategies_to_test = ["optimal", "conservative", "greedy"]
        player_counts = [2, 3, 4, 5]
        num_simulations = 500
        
        results_by_player_count = {}
        
        for player_count in player_counts:
            print(f"\nTesting with {player_count} players...")
            
            # Create balanced player configs (rotate strategies)
            player_configs = []
            for i in range(player_count):
                strategy = strategies_to_test[i % len(strategies_to_test)]
                player_configs.append(
                    (f"{strategy.capitalize()} {i+1}", PlayerType.AI_OPTIMAL, strategy)
                )
            
            result = simulator.simulate_games(
                player_configs=player_configs,
                num_simulations=num_simulations,
                target_score=6000
            )
            
            # Aggregate results by strategy
            strategy_wins = {}
            for name, rate in result.win_rates.items():
                strategy = name.split()[0].lower()
                strategy_wins[strategy] = strategy_wins.get(strategy, 0) + rate
            
            results_by_player_count[player_count] = strategy_wins
        
        # Print comparison table
        self._print_player_count_comparison(results_by_player_count, strategies_to_test)
    
    def _print_results(self, result, player_configs):
        """Print simulation results in a formatted table."""
        console = Console()
        
        # Create results table
        table = Table(title=f"Game Results ({result.num_simulations} games)")
        table.add_column("Player", style="cyan")
        table.add_column("Strategy", style="magenta")
        table.add_column("Win Rate", style="green")
        table.add_column("Avg Score", style="yellow")
        table.add_column("Games Won", style="blue")
        
        # Sort by win rate
        sorted_players = sorted(result.win_rates.items(), key=lambda x: x[1], reverse=True)
        
        for player_name, win_rate in sorted_players:
            # Find strategy name
            strategy = next(cfg[2] for cfg in player_configs if cfg[0] == player_name)
            games_won = int(win_rate * result.num_simulations)
            
            table.add_row(
                player_name,
                strategy,
                f"{win_rate:.1%}",
                f"{result.avg_scores[player_name]:.0f}",
                str(games_won)
            )
        
        console.print(table)
        console.print(f"\nAverage game length: {result.avg_rounds:.1f} rounds")
    
    def _create_strategy_results(self, result, player_configs, num_simulations) -> List[StrategyTestResult]:
        """Create detailed strategy results from simulation data."""
        strategy_results = []
        
        for player_name, win_rate in result.win_rates.items():
            strategy = next(cfg[2] for cfg in player_configs if cfg[0] == player_name)
            games_won = int(win_rate * num_simulations)
            
            # Calculate average winning score (when this player wins)
            winning_scores = result.score_distributions[player_name][
                result.score_distributions[player_name] >= 6000
            ]
            avg_winning_score = np.mean(winning_scores) if len(winning_scores) > 0 else 0
            
            strategy_results.append(StrategyTestResult(
                strategy_name=strategy,
                win_rate=win_rate,
                avg_score=result.avg_scores[player_name],
                avg_winning_score=avg_winning_score,
                avg_rounds_to_win=result.avg_rounds if games_won > 0 else 0,
                games_won=games_won,
                total_games=num_simulations
            ))
        
        return sorted(strategy_results, key=lambda x: x.win_rate, reverse=True)
    
    def _print_detailed_strategy_analysis(self, strategy_results: List[StrategyTestResult]):
        """Print detailed analysis of strategy performance."""
        console = Console()
        
        table = Table(title="Detailed Strategy Analysis")
        table.add_column("Strategy", style="cyan")
        table.add_column("Win Rate", style="green")
        table.add_column("Avg Score", style="yellow")
        table.add_column("Avg Win Score", style="magenta")
        table.add_column("Win Efficiency", style="blue")
        
        for result in strategy_results:
            # Calculate win efficiency (how much over 6000 they typically win by)
            win_efficiency = (result.avg_winning_score - 6000) / 6000 * 100 if result.avg_winning_score > 0 else 0
            
            table.add_row(
                result.strategy_name,
                f"{result.win_rate:.1%}",
                f"{result.avg_score:.0f}",
                f"{result.avg_winning_score:.0f}",
                f"{win_efficiency:.1f}%"
            )
        
        console.print(table)
    
    def _print_player_count_comparison(self, results_by_player_count, strategies):
        """Print comparison of strategy performance by player count."""
        console = Console()
        
        table = Table(title="Strategy Performance by Number of Players")
        table.add_column("Strategy", style="cyan")
        
        for player_count in sorted(results_by_player_count.keys()):
            table.add_column(f"{player_count} Players", style="green")
        
        for strategy in strategies:
            row = [strategy.capitalize()]
            for player_count in sorted(results_by_player_count.keys()):
                win_rate = results_by_player_count[player_count].get(strategy, 0)
                row.append(f"{win_rate:.1%}")
            table.add_row(*row)
        
        console.print(table)


if __name__ == "__main__":
    # Run tests directly
    test = TestStrategyComparison()
    simulator = GameSimulator(num_workers=4)
    
    print("Running strategy comparison tests...\n")
    
    # Run each test
    test.test_three_player_strategy_comparison(simulator)
    print("\n" + "="*80 + "\n")
    
    test.test_all_strategies_comparison(simulator)
    print("\n" + "="*80 + "\n")
    
    test.test_optimal_vs_probability_threshold(simulator)
    print("\n" + "="*80 + "\n")
    
    test.test_strategy_performance_by_player_count(simulator)