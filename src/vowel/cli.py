import importlib.util
import os
import sys

import dotenv

dotenv.load_dotenv()

if "--debug" in sys.argv:
    if importlib.util.find_spec("logfire"):
        import logfire

        logfire.configure()
        logfire.instrument_pydantic_ai()
    else:
        raise ImportError(
            "Debug mode enabled but logfire is not installed. Please install logfire or disable debug mode."
        )

from pathlib import Path

import click


@click.command()
@click.argument("yaml_file", type=click.Path(exists=True, path_type=Path))
@click.option("--debug", is_flag=True, help="Enable debug mode with logfire")
@click.option(
    "--filter", "-f", "filter_func", help="Only run specific function(s) (comma-separated)"
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output including reasons")
def main(yaml_file: Path, debug: bool, filter_func: str, verbose: bool):
    from .utils import import_function, run_evals

    filter_list = [f.strip() for f in filter_func.split(",")] if filter_func else None

    click.echo(f"Loaded evaluation(s) from: {click.style(yaml_file.name, fg='cyan')}")
    if debug:
        click.secho("🐛 Debug mode enabled (logfire active)", fg="yellow")

    try:
        summary = run_evals(yaml_file, filter_funcs=filter_list, debug=debug)
    except ValueError as e:
        click.secho(f"❌ {e}", fg="red", err=True)
        raise click.Abort()
    except Exception as e:
        click.secho(f"❌ Error running evaluations: {e}", fg="red", err=True)
        if debug:
            raise
        raise click.Abort()

    click.echo(f"\n{click.style('Available eval configurations:', bold=True)}")
    for result in summary.results:
        click.echo(f"  • {click.style(result.eval_id, fg='green')}")
    click.echo()

    for result in summary.results:
        click.echo(f"\n{'='*80}")
        click.echo(
            f"{click.style('Testing:', bold=True)} {click.style(result.eval_id, fg='cyan', bold=True)}"
        )
        click.echo("=" * 80)

        if result.error:
            click.secho(f"❌ Error: {result.error}", fg="red")
            continue

        if result.report:
            result.report.print(include_averages=True, include_reasons=True)

    click.echo("\n")

    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()

        summary_table = Table(
            title="📊 Final Summary", box=box.ROUNDED, show_header=True, header_style="bold cyan"
        )

        summary_table.add_column("Metric", style="cyan", no_wrap=True, width=20)
        summary_table.add_column("Count", justify="center", width=10)
        summary_table.add_column("Status", justify="center", width=10)

        summary_table.add_row("Total Functions", str(summary.total_count), "📊")

        summary_table.add_row(
            "Fully Passed",
            str(summary.success_count),
            "[green]✅[/green]" if summary.success_count > 0 else "−",
        )

        if summary.failed_count > 0:
            summary_table.add_row(
                "Partial Failures", str(summary.failed_count), "[yellow]⚠️[/yellow]"
            )

        if summary.error_count > 0:
            summary_table.add_row("Errors", str(summary.error_count), "[red]❌[/red]")

        console.print()
        console.print(summary_table)
        console.print()

    except ImportError:
        click.echo(f"  📊 Total functions: {summary.total_count}")
        click.echo(
            f"  ✅ Fully passed: {click.style(str(summary.success_count), fg='green', bold=True)}"
        )

        if summary.failed_count > 0:
            click.echo(
                f"  ⚠️  Partial failures: {click.style(str(summary.failed_count), fg='yellow', bold=True)}"
            )

        if summary.error_count > 0:
            click.echo(f"  ❌ Errors: {click.style(str(summary.error_count), fg='red', bold=True)}")

    if summary.failed_results:
        click.echo("\n")
        click.echo(click.style("⚠️  Functions with Failed Assertions", bold=True, fg="yellow"))
        click.echo("")

        for result in summary.failed_results:
            click.echo(f"\n🔍 {click.style(result.eval_id, fg='cyan', bold=True)}")

            for case in result.report.cases:
                failed_assertions = [
                    (name, res) for name, res in case.assertions.items() if not res.value
                ]

                if failed_assertions:
                    total_assertions = len(case.assertions)
                    failed_count = len(failed_assertions)

                    click.echo(f"\n  Case: {click.style(case.name, fg='yellow')}")
                    click.echo(
                        f"  Failed: {click.style(f'{failed_count}/{total_assertions}', fg='red')} assertions"
                    )

                    for assertion_name, res in failed_assertions:
                        click.echo(f"\n    ❌ {click.style(assertion_name, fg='red', bold=True)}")
                        if res.reason:
                            reason_lines = str(res.reason).split("\n")
                            for line in reason_lines:
                                if line.strip():
                                    click.echo(
                                        f"       {click.style(line.strip(), fg='red', dim=True)}"
                                    )

    click.echo()


if __name__ == "__main__":
    main()
