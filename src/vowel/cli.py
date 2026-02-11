"""Command-line interface for the vowel evaluation framework.

Usage:
    vowel run <yaml_file>             Run evaluations from a YAML spec
    vowel run -d <directory>          Run all YAML files in a directory
"""

import sys
import time
from pathlib import Path

import click
import dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from .utils import load_bundle, run_evals

dotenv.load_dotenv()
console = Console()


def find_yaml_files(directory: Path) -> list[Path]:
    """Find all YAML files in directory recursively."""
    return sorted((*directory.glob("**/*.yml"), *directory.glob("**/*.yaml")))


def validate_coverage(ctx, param, value):
    """Validate coverage is between 0 and 100."""
    new_value = max(1, min(100, value))
    click.secho(
        f"WARNING: Coverage expected to be in range between 1 and 100 but {int(value) if value == int(value) else value} found, defaulting to {new_value}.",
        fg="yellow",
        err=True,
    )
    return value


# ── Top-level group ───────────────────────────────────────────────


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """vowel — YAML-based evaluation framework for Python functions."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ── vowel run ─────────────────────────────────────────────────────


@main.command()
@click.argument("yaml_file", type=click.Path(exists=True, path_type=Path), required=False)
@click.option("--ci", is_flag=True, help="Enable CI mode")
@click.option(
    "--coverage",
    "--cov",
    type=float,
    default=100,
    help="Coverage percent",
    callback=validate_coverage,
)
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--filter", "-f", "filter_func", help="Filter functions (comma-separated)")
@click.option(
    "--dir",
    "-d",
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Run all YAML files in directory",
)
@click.option("--quiet", "-q", is_flag=True, help="Minimal output")
@click.option("--no-color", is_flag=True, help="Disable colors")
@click.option("--list-fixtures", is_flag=True, help="List fixtures")
@click.option("--dry-run", is_flag=True, help="Show test plan without running")
@click.option("--fixture-tree", is_flag=True, help="Show fixture tree")
@click.option("--export-json", type=click.Path(path_type=Path), help="Export results to JSON")
@click.option("--ignore-duration", is_flag=True, help="Ignore duration constraints")
@click.option("--watch", "-w", is_flag=True, help="Watch mode: re-run on file changes")
def run(
    yaml_file: Path | None,
    debug: bool,
    coverage: float,
    filter_func: str | None,
    directory: Path | None,
    quiet: bool,
    no_color: bool,
    ci: bool,
    list_fixtures: bool,
    dry_run: bool,
    fixture_tree: bool,
    export_json: Path | None,
    ignore_duration: bool,
    watch: bool,
):
    """Run evaluations from YAML spec files."""
    console = Console(force_terminal=False, no_color=True) if no_color else Console()

    # Validate incompatible options
    if directory and filter_func:
        click.secho("ERROR: --filter cannot be used with --dir", fg="red", err=True)
        click.secho(
            "Hint: When using --dir, all functions in all YAML files are evaluated.",
            fg="yellow",
            err=True,
        )
        raise SystemExit(1)

    # Handle --list-fixtures
    if list_fixtures:
        if not yaml_file:
            click.secho("ERROR: --list-fixtures requires a YAML file", fg="red", err=True)
            raise click.Abort()

        bundle = load_bundle(yaml_file)

        if not bundle.fixtures:
            console.print("[dim]No fixtures defined[/dim]")
            return

        # Fixtures table
        table = Table(title="Fixtures", box=box.ROUNDED)
        table.add_column("Name", style="cyan", header_style="bold cyan")
        table.add_column("Setup", style="dim")
        table.add_column("Teardown", style="dim")
        table.add_column("Scope", style="yellow")
        table.add_column("Used By", style="green")

        for name, defn in bundle.fixtures.items():
            users = []
            for eval_name, eval_def in bundle.evals.items():
                if eval_def.fixture and name in eval_def.fixture:
                    users.append(eval_name)

            table.add_row(
                name,
                defn.setup,
                defn.teardown or "[dim](none)[/dim]",
                defn.scope,
                ", ".join(users) if users else "[dim](none)[/dim]",
            )

        console.print()
        console.print(table)

        # Function -> Fixture relationships
        if bundle.evals:
            console.print()
            table2 = Table(title="Function -> Fixture Relationships", box=box.ROUNDED)
            table2.add_column("Function", style="green", header_style="bold green")
            table2.add_column("Fixtures", style="cyan")

            for eval_name, eval_def in bundle.evals.items():
                table2.add_row(
                    eval_name,
                    (", ".join(eval_def.fixture) if eval_def.fixture else "[dim](none)[/dim]"),
                )

            console.print(table2)
        return

    # Handle --dry-run
    if dry_run:
        if not yaml_file:
            click.secho("ERROR: --dry-run requires a YAML file", fg="red", err=True)
            raise click.Abort()

        bundle = load_bundle(yaml_file)

        filter_list = [f.strip() for f in filter_func.split(",")] if filter_func else None
        evals_to_run = bundle.evals

        if filter_list:
            evals_to_run = {k: v for k, v in bundle.evals.items() if k in filter_list}

        if not evals_to_run:
            console.print("[dim]No evals to run[/dim]")
            return

        total_cases = sum(len(e.dataset) for e in evals_to_run.values())

        # Info panel
        info = Table.grid(padding=0)
        info.add_column(style="cyan")
        info.add_column(style="dim")
        info.add_row(f"File: {yaml_file.name}", "")
        info.add_row(f"Fixtures: {len(bundle.fixtures)} defined", "")
        info.add_row(f"Functions: {len(bundle.evals)} defined", "")
        info.add_row(f"Total Test Cases: {total_cases}", "")

        console.print()
        console.print(Panel(info, title="Dry Run", border_style="blue"))

        # Test plan table
        console.print()
        table = Table(
            title="Test Plan",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold blue",
        )
        table.add_column("Function", style="cyan", no_wrap=True)
        table.add_column("Fixtures", style="yellow")
        table.add_column("Evaluators", style="green")
        table.add_column("Cases", justify="right", style="blue")

        for eval_name, eval_def in evals_to_run.items():
            case_count = len(eval_def.dataset)

            if eval_def.fixture:
                fixture_details = []
                for fn in eval_def.fixture:
                    if fn in bundle.fixtures:
                        scope = bundle.fixtures[fn].scope
                        fixture_details.append(f"{fn} ({scope})")
                    else:
                        fixture_details.append(f"[red]{fn} (undefined!)[/red]")
                fixtures_str = "\n".join(fixture_details)
            else:
                fixtures_str = "[dim](none)[/dim]"

            if eval_def.evals and isinstance(eval_def.evals, dict):
                evals_str = ", ".join(eval_def.evals.keys())
            else:
                evals_str = "[dim](none)[/dim]"

            table.add_row(eval_name, fixtures_str, evals_str, str(case_count))

        console.print(table)

        # Summary
        console.print()
        console.print(
            Panel(
                f"[cyan]{len(evals_to_run)}[/cyan] functions, [blue]{total_cases}[/blue] test cases",
                title="Summary",
                border_style="green",
            )
        )
        return

    # Handle --fixture-tree
    if fixture_tree:
        if not yaml_file:
            click.secho("ERROR: --fixture-tree requires a YAML file", fg="red", err=True)
            raise click.Abort()

        bundle = load_bundle(yaml_file)

        if not bundle.fixtures:
            console.print("[dim]No fixtures defined[/dim]")
            return

        fixture_users = {name: [] for name in bundle.fixtures}
        for eval_name, eval_def in bundle.evals.items():
            if eval_def.fixture:
                for fixture_name in eval_def.fixture:
                    if fixture_name in fixture_users:
                        fixture_users[fixture_name].append(eval_name)

        console.print()
        tree = Tree("Fixture Dependency Tree", highlight=True)

        for fixture_name in bundle.fixtures:
            branch = tree.add(f"[cyan bold]{fixture_name}[/cyan bold]")

            if fixture_users[fixture_name]:
                for user in fixture_users[fixture_name]:
                    user_fixtures = bundle.evals[user].fixture or []
                    other_fixtures = [f for f in user_fixtures if f != fixture_name]
                    fixture_str = (
                        f" [dim]+ {', '.join(other_fixtures)}[/dim]" if other_fixtures else ""
                    )
                    branch.add(f"[green]{user}[/green]{fixture_str}")
            else:
                branch.add("[dim](unused)[/dim]")

        console.print(tree)
        return

    # Handle --watch mode
    if watch:
        if not yaml_file:
            click.secho("ERROR: --watch requires a YAML file", fg="red", err=True)
            raise click.Abort()

        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            click.secho("ERROR: watchdog required for --watch", fg="red", err=True)
            click.secho("Install: pip install watchdog", fg="yellow", err=True)
            raise click.Abort() from None

        filter_list = [f.strip() for f in filter_func.split(",")] if filter_func else None
        watch_dir = yaml_file.parent

        def run_once():
            console.print()
            try:
                summary = run_evals(
                    yaml_file,
                    filter_funcs=filter_list,
                    debug=debug,
                    ignore_duration=ignore_duration,
                )
                for result in summary.results:
                    if result.error:
                        console.print(f"[red]Error: {result.eval_id}: {result.error}[/red]")
                    elif result.report:
                        result.report.print(include_averages=True, include_reasons=True)
                if summary.all_passed:
                    console.print(f"[green]✓ All {summary.total_count} passed[/green]")
                else:
                    console.print(
                        f"[yellow]⚠ {summary.success_count}/{summary.total_count} passed[/yellow]"
                    )
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

        class EvalHandler(FileSystemEventHandler):
            def __init__(self):
                self.last_run = 0

            def on_modified(self, event):
                if event.is_directory:
                    return
                now = time.time()
                if now - self.last_run < 1.0:
                    return
                self.last_run = now
                console.print(f"\n[cyan]Changed: {Path(event.src_path).name}[/cyan]")
                console.print("[dim]─[/dim]" * 40)
                run_once()

        console.print(f"[cyan]👁 Watch: {yaml_file.name}[/cyan]")
        console.print("[dim]Ctrl+C to stop[/dim]")
        console.print("[dim]─[/dim]" * 40)
        run_once()

        handler = EvalHandler()
        observer = Observer()
        observer.schedule(handler, str(watch_dir), recursive=False)
        observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            console.print("\n[yellow]Watch stopped[/yellow]")
        observer.join()
        return

    # Determine YAML files to run
    if directory:
        yaml_files = find_yaml_files(directory)
        if not yaml_files:
            click.secho(f"ERROR: No YAML files found in {directory}", fg="red", err=True)
            raise click.Abort()
        if not quiet:
            console.print(f"Found [cyan]{len(yaml_files)}[/cyan] YAML file(s)")
    elif yaml_file:
        yaml_files = [yaml_file]
    else:
        click.secho("ERROR: Either YAML_FILE or --dir is required", fg="red", err=True)
        raise click.Abort()

    filter_list = [f.strip() for f in filter_func.split(",")] if filter_func else None

    all_summaries = []
    for yf in yaml_files:
        try:
            summary = run_evals(
                yf,
                filter_funcs=filter_list,
                debug=debug,
                ignore_duration=ignore_duration,
            )
        except ValueError as e:
            click.secho(f"ERROR: {e}", fg="red", err=True)
            if not directory:
                raise SystemExit(1) from None
            continue
        except Exception as e:
            click.secho(f"ERROR: {e}", fg="red", err=True)
            if debug:
                raise
            if not directory:
                raise SystemExit(1) from None
            continue

        all_summaries.append((yf, summary))

    if not all_summaries:
        click.secho("ERROR: No evaluations completed", fg="red", err=True)
        raise click.Abort()

    # Multiple files mode
    if directory and len(all_summaries) > 1:
        total_functions = 0
        total_success = 0
        total_failed = 0
        total_errors = 0

        for yf, summary in all_summaries:
            if not quiet:
                console.print()
                console.print(Panel(f"[cyan]{yf.name}[/cyan]", border_style="cyan"))

                for result in summary.results:
                    if result.error:
                        console.print(f"  [red]x {result.eval_id}[/red]: {result.error}")
                    elif result.report:
                        console.print()
                        console.print(Panel(result.eval_id, border_style="blue"))
                        result.report.print(include_averages=True, include_reasons=True)

            total_functions += summary.total_count
            total_success += summary.success_count
            total_failed += summary.failed_count
            total_errors += summary.error_count

        if not quiet:
            console.print()
            console.print(Panel("Combined Summary", border_style="blue"))
            console.print(f"  Total files: [cyan]{len(all_summaries)}[/cyan]")
            console.print(f"  Total functions: [cyan]{total_functions}[/cyan]")
            console.print(f"  Passed: [green]{total_success}[/green]")
            if total_failed > 0:
                console.print(f"  Failed: [yellow]{total_failed}[/yellow]")
            if total_errors > 0:
                console.print(f"  Errors: [red]{total_errors}[/red]")

        if ci:
            failed_coverage = False
            for yf, summary in all_summaries:
                for result in summary.results:
                    if result.report:
                        assertion_percent = 100 * result.report.averages().assertions
                        if assertion_percent < coverage:
                            click.secho(
                                f"ERROR: [{yf.name}] '{result.eval_id}' failed: {assertion_percent:.1f}% < {coverage:.1f}%",
                                fg="red",
                                bold=True,
                                err=True,
                            )
                            failed_coverage = True

            if failed_coverage:
                raise SystemExit(1)
        return

    # Single file mode
    summary = all_summaries[0][1]

    if not quiet:
        console.print()

    for result in summary.results:
        if result.error:
            console.print(f"[red]Error: {result.eval_id}: {result.error}[/red]")
            continue

        if result.report:
            result.report.print(include_averages=True, include_reasons=True)

    console.print()

    # Summary table
    summary_table = Table(
        title="Summary", box=box.ROUNDED, show_header=True, header_style="bold cyan"
    )
    summary_table.add_column("Metric", style="cyan", no_wrap=True, width=20)
    summary_table.add_column("Count", justify="center", width=10)
    summary_table.add_column("Status", justify="center", width=15)

    summary_table.add_row("Total Functions", str(summary.total_count), "[blue]-[/blue]")

    summary_table.add_row(
        "Passed",
        str(summary.success_count),
        "[green]PASS[/green]" if summary.success_count > 0 else "[dim]-[/dim]",
    )

    if summary.failed_count > 0:
        summary_table.add_row("Failed", str(summary.failed_count), "[yellow]FAIL[/yellow]")

    if summary.error_count > 0:
        summary_table.add_row("Errors", str(summary.error_count), "[red]ERROR[/red]")

    console.print(summary_table)

    # Export JSON
    if export_json:
        import json

        json_data = summary.json()
        with open(export_json, "w") as f:
            json.dump(json_data, f, indent=2)
        if not quiet:
            console.print(f"[green]Results exported to: {export_json}[/green]")

    # Failed assertions detail
    if summary.failed_results:
        console.print()
        for result in summary.failed_results:
            console.print(Panel(result.eval_id, title="Failed Assertions", border_style="yellow"))

            for case in result.report.cases:
                failed_assertions = [
                    (name, res) for name, res in case.assertions.items() if not res.value
                ]

                if failed_assertions:
                    total_assertions = len(case.assertions)
                    failed_count = len(failed_assertions)

                    console.print(f"\n  [yellow]Case: {case.name}[/yellow]")
                    console.print(
                        f"  Failed: [red]{failed_count}/{total_assertions}[/red] assertions"
                    )

                    for assertion_name, res in failed_assertions:
                        console.print(f"\n    [red]x {assertion_name}[/red]")
                        if res.reason:
                            reason_lines = str(res.reason).split("\n")
                            for line in reason_lines:
                                if line.strip():
                                    console.print(f"       [dim]{line.strip()}[/dim]")

    console.print()

    # CI mode
    if ci:
        if coverage == 100 and not summary.all_passed:
            click.secho("ERROR: Some evaluations failed!", fg="red", bold=True, err=True)
            print(
                "::error title=Evaluation Failed::Some evaluations failed.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        else:
            for result in summary.results:
                if result.report:
                    assertion_percent = 100 * result.report.averages().assertions
                    if assertion_percent < coverage:
                        msg = f"ERROR: Coverage {assertion_percent:.1f}% < {coverage:.1f}% for '{result.eval_id}'"
                        click.secho(msg, fg="red", bold=True, err=True)
                        raise SystemExit(1)


if __name__ == "__main__":
    main()
