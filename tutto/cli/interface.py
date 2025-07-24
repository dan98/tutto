import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.progress import track
from typing import List, Optional
from ..core.game import Game, Round, RoundStatus
from ..core.dice import DiceRoll
from ..simulation import GameSimulator
from ..simulation.strategy_registry import strategy_registry


console = Console()


class InteractiveCLI:
    """Interactive command-line interface for Tutto."""
    
    def __init__(self):
        self.game = Game()
        self.simulator = GameSimulator()
        self.current_round: Optional[Round] = None
        self.current_strategy = "optimal"  # Default strategy for simulations
        self.recommendation_strategy = "optimal"  # Strategy for recommendations during play
    
    def display_roll(self, roll: DiceRoll, score: int, combinations: list):
        """Display dice roll results."""
        table = Table(title="Dice Roll Results")
        table.add_column("Dice", style="cyan")
        table.add_column("Values", style="magenta")
        
        dice_display = " ".join([f"[{v}]" for v in roll.values])
        table.add_row("Rolled", dice_display)
        
        console.print(table)
        
        if combinations:
            console.print(f"\n[green]Scoring combinations found:[/green]")
            for combo in combinations:
                console.print(f"  • {combo}")
        else:
            console.print("[red]No scoring dice! BUSTED![/red]")
        
        console.print(f"\n[yellow]Total possible score from this roll: {score}[/yellow]")
    
    def display_round_status(self, round_info: dict):
        """Display current round status."""
        from ..core.cards import CardType
        
        if round_info['card_type'] == CardType.STRASSE:
            collected = sorted(round_info['strasse_collected'])
            needed = sorted(set(range(1, 7)) - round_info['strasse_collected'])
            
            panel_content = (
                f"[cyan]Card Type:[/cyan] STRASSE (2000 points)\n"
                f"[green]Collected Values:[/green] {collected if collected else 'None'}\n"
                f"[red]Still Need:[/red] {needed if needed else 'Complete!'}\n"
                f"[yellow]Remaining Dice:[/yellow] {round_info['remaining_dice']}\n"
                f"[bold]Status:[/bold] {'COMPLETE!' if round_info['strasse_complete'] else 'In Progress'}"
            )
            title = "STRASSE Status"
        else:
            panel_content = (
                f"[cyan]Bonus Card:[/cyan] {round_info['bonus_card']}\n"
                f"[green]Accumulated Score:[/green] {round_info['accumulated_score']}\n"
                f"[yellow]Remaining Dice:[/yellow] {round_info['remaining_dice']}\n"
                f"[magenta]Tutto Achieved:[/magenta] {'Yes' if round_info['achieved_tutto'] else 'No'}\n"
                f"[bold]Total if stopped now:[/bold] {round_info['total_score_if_stop']}"
            )
            title = "Round Status"
        
        console.print(Panel(panel_content, title=title, border_style="blue"))
    
    def get_dice_input(self, num_dice: int) -> List[int]:
        """Get dice values from user input."""
        console.print(f"\n[cyan]Enter {num_dice} dice values (1-6):[/cyan]")
        values = []
        
        for i in range(num_dice):
            while True:
                value = IntPrompt.ask(f"Die {i+1}", default=0)
                if 1 <= value <= 6:
                    values.append(value)
                    break
                console.print("[red]Please enter a value between 1 and 6[/red]")
        
        return values
    
    def select_dice_to_keep(self, roll: DiceRoll) -> List[int]:
        """Interactive dice selection."""
        console.print("\n[cyan]Select dice to keep:[/cyan]")
        console.print("Current roll:", " ".join([f"[{v}]" for v in roll.values]))
        
        selected = []
        available_dice = list(roll.values)
        
        while True:
            console.print(f"\nSelected so far: {selected if selected else 'None'}")
            console.print(f"Available: {available_dice}")
            
            if not available_dice:
                break
            
            value = IntPrompt.ask("Select a die value to keep (0 to finish)", default=0)
            
            if value == 0:
                if not selected:
                    console.print("[red]You must select at least one scoring die![/red]")
                    continue
                break
            
            if value in available_dice:
                selected.append(value)
                available_dice.remove(value)
            else:
                console.print(f"[red]Die with value {value} not available[/red]")
        
        return selected
    
    def show_simulation_results(self, sim_result):
        """Display simulation results."""
        console.print("\n[bold cyan]Simulation Results:[/bold cyan]")
        console.print(sim_result)
        
        # Create a simple histogram showing individual scores
        console.print("\n[bold]Score Distribution (if continuing from this point):[/bold]")
        
        # Count occurrences of each score
        score_counts = {}
        for score in sim_result.score_distribution:
            score_counts[int(score)] = score_counts.get(int(score), 0) + 1
        
        # Special case: all scores are the same
        if len(score_counts) == 1:
            score = list(score_counts.keys())[0]
            console.print(f"All simulations resulted in score: {score}")
            console.print(f"{score:5d}: {'█' * 40} {len(sim_result.score_distribution):5d} (100.0%)")
            return
        
        # Find max count for scaling bars
        max_count = max(score_counts.values())
        
        # Display histogram - show top scores by frequency
        sorted_scores = sorted(score_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Show all scores if there are 20 or fewer, otherwise show top 20
        scores_to_show = sorted_scores if len(sorted_scores) <= 20 else sorted_scores[:20]
        
        # Re-sort by score value for display
        scores_to_show.sort(key=lambda x: x[0])
        
        for score, count in scores_to_show:
            bar_length = int(40 * count / max_count)
            bar = "█" * bar_length if bar_length > 0 else ""
            percentage = (count / len(sim_result.score_distribution)) * 100
            console.print(f"{score:5d}: {bar:<40} {count:5d} ({percentage:5.2f}%)")
        
        if len(sorted_scores) > 20:
            console.print(f"\n[dim](Showing top 20 most frequent scores out of {len(sorted_scores)} unique values)[/dim]")
    
    def play_round(self):
        """Play a single round interactively."""
        # Ask for card type
        from ..core.cards import CardType
        
        card_type_choice = Prompt.ask(
            "\n[cyan]Card type[/cyan]",
            choices=["bonus", "strasse"],
            default="bonus"
        )
        
        if card_type_choice == "strasse":
            card_type = CardType.STRASSE
            bonus_value = 2000
            console.print(f"\n[bold green]Starting STRASSE round - collect 1,2,3,4,5,6 to win 2000 points![/bold green]")
        else:
            card_type = CardType.BONUS
            bonus_value = IntPrompt.ask("[cyan]Enter bonus value[/cyan]", default=300)
            console.print(f"\n[bold green]Starting BONUS round with {bonus_value} point bonus[/bold green]")
        
        self.current_round = self.game.start_new_round(card_type=card_type, bonus_value=bonus_value)
        console.print(f"[dim]Using {self.recommendation_strategy} strategy for recommendations[/dim]")
        
        while self.current_round.state.status == RoundStatus.IN_PROGRESS:
            round_info = self.current_round.get_decision_info()
            self.display_round_status(round_info)
            
            # Ask whether to use manual or random dice
            use_manual = Confirm.ask("\n[cyan]Enter dice values manually?[/cyan]", default=False)
            
            if use_manual:
                dice_values = self.get_dice_input(self.current_round.dice.remaining_dice)
                roll, score, combinations = self.current_round.roll_dice(dice_values)
            else:
                roll, score, combinations = self.current_round.roll_dice()
            
            self.display_roll(roll, score, combinations)
            
            if self.current_round.state.status == RoundStatus.BUSTED:
                console.print("\n[bold red]BUSTED! Round ends with 0 points.[/bold red]")
                break
            
            # Handle dice selection based on card type
            if self.current_round.state.card_type == CardType.STRASSE:
                # For Strasse, automatically select new values
                new_values = set(roll.values) - self.current_round.state.strasse_collected
                
                if new_values:
                    # Keep one die per new value
                    selected_dice = []
                    for value in sorted(new_values):
                        # Find first occurrence of this value
                        for i, v in enumerate(roll.values):
                            if v == value and v not in selected_dice:
                                selected_dice.append(v)
                                break
                    
                    console.print(f"\n[cyan]New values found: {sorted(new_values)}[/cyan]")
                    console.print(f"[green]Automatically keeping one die per new value: {selected_dice}[/green]")
                else:
                    console.print("\n[bold red]No new values found![/bold red]")
                    selected_dice = []
            else:
                # Regular scoring logic
                strategy = strategy_registry.get_strategy(self.recommendation_strategy)
                recommended_dice = strategy.select_dice(roll, self.current_round.scoring_engine)
                
                if recommended_dice:
                    # Show recommendation
                    console.print(f"\n[bold cyan]{self.recommendation_strategy.capitalize()} Strategy Recommendation:[/bold cyan]")
                    console.print(f"[yellow]Keep these dice: {recommended_dice}[/yellow]")
                    
                    # Calculate score for recommended dice
                    recommended_roll = DiceRoll(recommended_dice)
                    rec_score, _ = self.current_round.scoring_engine.calculate_score(recommended_roll)
                    console.print(f"[yellow]This selection scores: {rec_score} points[/yellow]")
                    
                    # Ask if user wants to use automatic selection
                    use_auto = Confirm.ask("\n[cyan]Use recommended selection?[/cyan]", default=True)
                    
                    if use_auto:
                        selected_dice = recommended_dice
                        console.print(f"[green]Automatically selected: {selected_dice}[/green]")
                    else:
                        # Manual selection
                        selected_dice = self.select_dice_to_keep(roll)
                else:
                    console.print("\n[bold red]No scoring dice available![/bold red]")
                    selected_dice = []
            
            score_gained = self.current_round.select_dice(selected_dice)
            console.print(f"\n[green]Added {score_gained} points to your score![/green]")
            
            if self.current_round.state.achieved_tutto:
                console.print("\n[bold magenta]TUTTO! All dice used successfully![/bold magenta]")
                console.print(f"[bold magenta]Bonus {bonus_value} will be added to your score![/bold magenta]")
            
            # Run simulation for decision support (skip for Strasse - no decisions)
            if self.current_round.dice.remaining_dice > 0 and self.current_round.state.card_type == CardType.BONUS:
                remaining = self.current_round.dice.remaining_dice
                console.print(f"\n[cyan]Running simulation to analyze continuation with {remaining} {'die' if remaining == 1 else 'dice'} remaining...[/cyan]")
                
                with console.status("[bold green]Simulating 100,000 games..."):
                    sim_result = self.simulator.simulate_continuation(
                        self.current_round.get_decision_info(),
                        num_simulations=100000,
                        strategy=self.current_strategy
                    )
                
                self.show_simulation_results(sim_result)
                
                # Show strategy decision
                strategy = strategy_registry.get_strategy(self.recommendation_strategy)
                strategy_continue = strategy.should_continue(self.current_round.get_decision_state())
                
                console.print(f"\n[bold cyan]{self.recommendation_strategy.capitalize()} Strategy Recommendation:[/bold cyan]")
                
                if self.recommendation_strategy == "optimal":
                    # For optimal strategy, show expected value reasoning
                    if strategy_continue:
                        console.print(f"[green]CONTINUE - Expected value ({sim_result.expected_value:.1f}) > Current score ({self.current_round.state.total_score})[/green]")
                    else:
                        console.print(f"[red]STOP - Current score ({self.current_round.state.total_score}) > Expected value ({sim_result.expected_value:.1f})[/red]")
                elif self.recommendation_strategy == "conservative":
                    # For conservative, show threshold reasoning
                    if strategy_continue:
                        console.print(f"[green]CONTINUE - Current score ({self.current_round.state.accumulated_score}) < threshold (300)[/green]")
                    else:
                        console.print(f"[red]STOP - Current score ({self.current_round.state.accumulated_score}) >= threshold (300)[/red]")
                elif self.recommendation_strategy == "greedy":
                    console.print(f"[green]CONTINUE - Greedy strategy always continues while dice remain[/green]")
                else:
                    # Generic message for other strategies
                    if strategy_continue:
                        console.print(f"[green]CONTINUE - Strategy recommends continuing[/green]")
                    else:
                        console.print(f"[red]STOP - Strategy recommends stopping[/red]")
                
                # Ask whether to continue
                remaining = self.current_round.dice.remaining_dice
                if Confirm.ask(f"\n[yellow]Continue rolling {remaining} {'die' if remaining == 1 else 'dice'}?[/yellow]", default=strategy_continue):
                    continue
                else:
                    self.current_round.stop_round()
                    break
            else:
                if self.current_round.state.card_type == CardType.STRASSE:
                    # For Strasse, automatically stop if we've collected all values
                    if self.current_round.state.strasse_complete:
                        console.print("\n[bold green]STRASSE COMPLETE! You win 2000 points![/bold green]")
                        break
                    else:
                        # In Strasse, you must continue - no choice
                        console.print("\n[yellow]No dice remaining - rolling all 6 dice again...[/yellow]")
                        console.print("[dim]In Strasse, you must continue until you collect all values or bust[/dim]")
                        # Auto-continue after a brief pause
                        continue
                else:
                    console.print("\n[yellow]No dice remaining - must roll all 6 dice again![/yellow]")
                    if not Confirm.ask("Continue?", default=True):
                        self.current_round.stop_round()
                        break
        
        # Complete round
        self.game.complete_round()
        final_score = self.current_round.state.total_score
        console.print(f"\n[bold green]Round completed! Final score: {final_score}[/bold green]")
        
        # Show game stats
        stats = self.game.get_game_stats()
        console.print(f"\n[cyan]Total game score: {stats['total_score']}[/cyan]")
    
    def run(self):
        """Main CLI loop."""
        console.print(Panel.fit(
            "[bold cyan]Welcome to Tutto![/bold cyan]\n"
            "A dice game simulator with probability analysis",
            border_style="blue"
        ))
        
        while True:
            choice = Prompt.ask(
                "\n[cyan]What would you like to do?[/cyan]",
                choices=["play", "simulate", "stats", "strategy", "recommendation", "quit"],
                default="play"
            )
            
            if choice == "play":
                self.play_round()
            elif choice == "simulate":
                self.run_pure_simulation()
            elif choice == "stats":
                self.show_game_stats()
            elif choice == "strategy":
                self.select_strategy()
            elif choice == "recommendation":
                self.select_recommendation_strategy()
            elif choice == "quit":
                console.print("[yellow]Thanks for playing![/yellow]")
                break
    
    def run_pure_simulation(self):
        """Run simulation without playing."""
        from ..core.cards import CardType
        
        card_type_choice = Prompt.ask(
            "[cyan]Card type[/cyan]",
            choices=["bonus", "strasse"],
            default="bonus"
        )
        
        if card_type_choice == "strasse":
            card_type = CardType.STRASSE
            bonus = 2000
            console.print("\n[bold]Strasse Simulation[/bold]")
            console.print("Simulating probability of completing 1,2,3,4,5,6 for 2000 points")
            
            # For Strasse, ask how many values already collected
            collected_str = Prompt.ask(
                "[cyan]Already collected values (comma-separated, e.g., '1,2,4')[/cyan]",
                default=""
            )
            if collected_str:
                collected = set(int(x.strip()) for x in collected_str.split(",") if x.strip())
            else:
                collected = set()
            
            remaining_dice = IntPrompt.ask("[cyan]Remaining dice[/cyan]", default=6)
            
            state = {
                "card_type": card_type,
                "bonus_card": bonus,
                "accumulated_score": 0,
                "remaining_dice": remaining_dice,
                "achieved_tutto": False,
                "round_status": "in_progress",
                "total_score_if_stop": 0,
                "strasse_collected": collected,
                "strasse_complete": len(collected) == 6,
            }
        else:
            card_type = CardType.BONUS
            bonus = IntPrompt.ask("[cyan]Bonus card value[/cyan]", default=300)
            accumulated = IntPrompt.ask("[cyan]Current accumulated score[/cyan]", default=0)
            remaining_dice = IntPrompt.ask("[cyan]Remaining dice[/cyan]", default=6)
            
            state = {
                "card_type": card_type,
                "bonus_card": bonus,
                "accumulated_score": accumulated,
                "remaining_dice": remaining_dice,
                "achieved_tutto": False,
                "round_status": "in_progress",
                "total_score_if_stop": accumulated,
            }
        
        console.print("\n[cyan]Running simulation...[/cyan]")
        sim_result = self.simulator.simulate_continuation(
            state, 
            num_simulations=100000,
            strategy=self.current_strategy
        )
        self.show_simulation_results(sim_result)
        
        if card_type == CardType.STRASSE:
            # For Strasse, show specific probabilities
            console.print(f"\n[bold]Strasse Statistics:[/bold]")
            console.print(f"  Need: {sorted(set(range(1,7)) - state['strasse_collected'])}")
            console.print(f"  Success rate: {sim_result.success_rate:.2%}")
            console.print(f"  Expected value: {sim_result.expected_value:.1f} points")
        else:
            # Show probability calculations for regular dice
            probs = self.simulator.calculate_probabilities(remaining_dice)
            console.print(f"\n[bold]Probabilities with {remaining_dice} dice:[/bold]")
            console.print(f"  Scoring: {probs['scoring_probability']:.1%}")
            console.print(f"  Bust: {probs['bust_probability']:.1%}")
            console.print(f"  Tutto possible: {probs['tutto_possibility']:.1%}")
    
    def show_game_stats(self):
        """Display game statistics."""
        stats = self.game.get_game_stats()
        
        table = Table(title="Game Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in stats.items():
            if isinstance(value, float):
                table.add_row(key.replace("_", " ").title(), f"{value:.1f}")
            else:
                table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(table)


    def select_strategy(self):
        """Allow user to select and configure a strategy."""
        # Show available strategies
        strategies_info = strategy_registry.get_all_strategies_info()
        
        table = Table(title="Available Strategies")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Parameters", style="yellow")
        
        for name, config in strategies_info.items():
            params = ", ".join([f"{k}={v}" for k, v in config.parameters.items()])
            table.add_row(name, config.description, params or "None")
        
        console.print(table)
        console.print(f"\n[bold]Current strategy: {self.current_strategy}[/bold]")
        
        # Ask for strategy selection
        strategy_name = Prompt.ask(
            "\n[cyan]Select strategy[/cyan]",
            choices=list(strategies_info.keys()),
            default=self.current_strategy
        )
        
        self.current_strategy = strategy_name
        console.print(f"\n[green]Strategy set to: {strategy_name}[/green]")
        
        # Show strategy-specific configuration if needed
        config = strategies_info[strategy_name]
        if config.parameters:
            console.print("\n[cyan]Strategy parameters:[/cyan]")
            for param, default_value in config.parameters.items():
                console.print(f"  {param}: {default_value}")

    def select_recommendation_strategy(self):
        """Select strategy to use for recommendations during play."""
        strategies_info = strategy_registry.get_all_strategies_info()
        
        table = Table(title="Available Recommendation Strategies")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        
        for name, config in strategies_info.items():
            table.add_row(name, config.description)
        
        console.print(table)
        console.print(f"\n[bold]Current recommendation strategy: {self.recommendation_strategy}[/bold]")
        console.print(f"[dim]Simulation strategy: {self.current_strategy}[/dim]")
        
        # Ask for strategy selection
        strategy_name = Prompt.ask(
            "\n[cyan]Select recommendation strategy for gameplay[/cyan]",
            choices=list(strategies_info.keys()),
            default=self.recommendation_strategy
        )
        
        self.recommendation_strategy = strategy_name
        console.print(f"\n[green]Recommendation strategy set to: {strategy_name}[/green]")
        console.print(f"[dim]This will be used to suggest moves during play[/dim]")


# Import numpy here to avoid circular imports
import numpy as np