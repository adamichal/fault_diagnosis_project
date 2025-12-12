from __future__ import annotations

from experiments.reporting import (
    print_markdown_diagnosis_table,
    print_markdown_reconstruction_table,
)
from experiments.runner import run_multiple_benchmarks


def main():
    configs = [

    ]

    results = run_multiple_benchmarks(configs)
    print_markdown_diagnosis_table(results)
    print_markdown_reconstruction_table(results, which="union")


if __name__ == "__main__":
    main()
