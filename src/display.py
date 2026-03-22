"""
Shared display utilities for consistent CLI output formatting.

Kept the comments minimal as the function names mostly speak for themselves.
"""

# Standard width for dividers
WIDTH = 50


def divider(char: str = "=") -> str:
    return char * WIDTH


def print_divider(char: str = "=") -> None:
    print(divider(char))


def print_header(title: str) -> None:
    print(divider("="))
    print(title)
    print(divider("="))


def print_subheader(title: str) -> None:
    print(divider("-"))
    print(title)
    print(divider("-"))


def print_epoch_header(epoch: int, total_epochs: int) -> None:
    print(f"\nEpoch {epoch}/{total_epochs}")
    print(divider("-"))


def print_section(title: str) -> None:
    print()
    print(divider("="))
    print(title)
    print(divider("="))


def print_results_header(title: str = "RESULTS") -> None:
    print("\n" + divider("="))
    print(title)
    print(divider("="))


def print_metrics(metrics: dict[str, float], indent: int = 0) -> None:
    prefix = " " * indent
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{prefix}{name}: {value:.4f}")
        else:
            print(f"{prefix}{name}: {value}")
