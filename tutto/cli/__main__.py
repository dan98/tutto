import click
from .interface import InteractiveCLI


@click.command()
@click.option('--simulate', '-s', is_flag=True, help='Run in simulation mode')
@click.option('--rounds', '-r', type=int, help='Number of rounds to simulate')
def main(simulate, rounds):
    """Tutto - A dice game simulator with probability analysis."""
    cli = InteractiveCLI()
    
    if simulate and rounds:
        # Run automated simulation
        click.echo(f"Running {rounds} rounds of simulation...")
        # TODO: Implement batch simulation mode
    else:
        # Run interactive mode
        cli.run()


if __name__ == "__main__":
    main()