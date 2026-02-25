"""Command-line interface for the vowel evaluation framework.

Usage:
    vowel <yaml_file>                 Run evaluations from a YAML spec
    vowel -d <directory>              Run all YAML files in a directory
    vowel <yaml_file> -v              Detailed summary with spec semantics
    vowel <yaml_file> --hide-report   Hide pydantic_evals report output
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

from .eval_types import (
    AssertionCase,
    ContainsInputCase,
    DurationCase,
    IsInstanceCase,
    LLMJudgeCase,
    PatternMatchCase,
)
from .utils import EvalsBundle, EvalSummary, load_bundle, run_evals

dotenv.load_dotenv()
console = Console()


def _eval_type_label(case) -> str:
    """Return short human label for an evaluator case type."""
    labels = {
        IsInstanceCase: "Type",
        AssertionCase: "Assertion",
        DurationCase: "Duration",
        ContainsInputCase: "ContainsInput",
        PatternMatchCase: "Pattern",
        LLMJudgeCase: "LLMJudge",
    }
    return labels.get(type(case), type(case).__name__)


def _print_verbose_summary(
    console: Console, summary: EvalSummary, bundle: EvalsBundle, yaml_file: Path
) -> None:
    """Print a detailed evaluation summary with spec semantics and result breakdown."""
    total_cases = sum(len(e.dataset) for e in bundle.evals.values())
    total_global_evals = sum(len(e.evals) for e in bundle.evals.values())

    # ── Spec Overview Panel ──
    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold")
    info.add_column()
    info.add_row("File", str(yaml_file.name))
    info.add_row("Functions", str(len(bundle.evals)))
    info.add_row("Total Cases", str(total_cases))
    info.add_row("Global Evaluators", str(total_global_evals))
    info.add_row("Fixtures", str(len(bundle.fixtures)) if bundle.fixtures else "none")
    if bundle.fixtures:
        for fname, fdef in bundle.fixtures.items():
            setup_label = fdef.cls or fdef.setup or "none"
            kind = "cls" if fdef.cls else "setup"
            teardown_label = fdef.teardown or "none"
            info.add_row(
                f"  {fname}", f"scope={fdef.scope}  {kind}={setup_label}  teardown={teardown_label}"
            )
    console.print()
    console.print(Panel(info, title="Spec Overview", border_style="bright_cyan"))

    # ── Per-Function Detail Table ──
    detail = Table(
        title="Evaluation Detail",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white",
        expand=True,
    )
    detail.add_column("Function", style="white bold", no_wrap=True)
    detail.add_column("Fixtures", style="grey70")
    detail.add_column("Global Evaluators", style="grey70")
    detail.add_column("Cases", justify="center", style="white")
    detail.add_column("Passed", justify="center", style="green")
    detail.add_column("Failed", justify="center", style="bright_red")
    detail.add_column("Coverage", justify="center")

    for result in summary.results:
        eval_def = bundle.evals.get(result.eval_id)
        if not eval_def:
            detail.add_row(result.eval_id, "-", "-", "-", "-", "-", "-")
            continue

        fixtures_str = ", ".join(eval_def.fixture) if eval_def.fixture else "[grey50]none[/grey50]"

        # Global evaluators with types
        if eval_def.evals:
            eval_parts = []
            for ename, ecase in eval_def.evals.items():
                eval_parts.append(f"{ename} [grey50]({_eval_type_label(ecase)})[/grey50]")
            evals_str = "\n".join(eval_parts)
        else:
            evals_str = "[grey50]none[/grey50]"

        case_count = len(eval_def.dataset)

        if result.error:
            detail.add_row(
                result.eval_id,
                fixtures_str,
                evals_str,
                str(case_count),
                "-",
                "-",
                "[bright_red]ERROR[/bright_red]",
            )
        elif result.report:
            passed = sum(
                1 for c in result.report.cases if all(a.value for a in c.assertions.values())
            )
            failed = len(result.report.cases) - passed
            cov = passed / len(result.report.cases) * 100 if result.report.cases else 100
            cov_style = "green" if cov == 100 else "yellow" if cov >= 50 else "red"
            detail.add_row(
                result.eval_id,
                fixtures_str,
                evals_str,
                str(case_count),
                str(passed),
                str(failed) if failed else "[grey50]0[/grey50]",
                f"[{cov_style}]{cov:.0f}%[/{cov_style}]",
            )

    console.print()
    console.print(detail)

    # ── Case-Level Breakdown ──
    for result in summary.results:
        if not result.report:
            continue
        eval_def = bundle.evals.get(result.eval_id)
        if not eval_def:
            continue

        case_table = Table(
            title=f"Cases: {result.eval_id}",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold white",
        )
        case_table.add_column("Case", style="white", no_wrap=True)
        case_table.add_column("Expected", style="grey70", max_width=25, overflow="ellipsis")
        case_table.add_column("Assertions", style="grey70")
        case_table.add_column("Status", justify="center")

        for i, case_result in enumerate(result.report.cases):
            case_name = case_result.name or f"case_{i}"
            total_assertions = len(case_result.assertions)
            passed_assertions = sum(1 for a in case_result.assertions.values() if a.value)
            all_pass = passed_assertions == total_assertions

            # Get expected from dataset if available
            ds_case = eval_def.dataset[i].case if i < len(eval_def.dataset) else None
            expected_str = "-"
            if ds_case:
                if ds_case.raises:
                    expected_str = ds_case.raises
                elif ds_case.has_expected:
                    expected_str = repr(ds_case.expected)

            status = (
                "[green]PASS[/green]"
                if all_pass
                else f"[bright_red]{passed_assertions}/{total_assertions}[/bright_red]"
            )

            # Build assertion names only (strip expression after colon)
            assertion_parts = []
            for aname, ares in case_result.assertions.items():
                short_name = aname.split(":")[0].strip()
                mark = "[green]✓[/green]" if ares.value else "[bright_red]✗[/bright_red]"
                assertion_parts.append(f"{mark} {short_name}")
            assertions_str = ", ".join(assertion_parts)

            case_table.add_row(case_name, expected_str, assertions_str, status)

        console.print()
        console.print(case_table)

    # ── Overall Summary ──
    _print_overall_summary(console, summary)


def _print_overall_summary(console: Console, summary: EvalSummary) -> None:
    """Print the Overall Summary panel (used in both default and verbose modes)."""
    total_case_count = sum(len(r.report.cases) for r in summary.results if r.report)
    passed_case_count = sum(
        1
        for r in summary.results
        if r.report
        for c in r.report.cases
        if all(a.value for a in c.assertions.values())
    )
    failed_case_count = total_case_count - passed_case_count
    overall_cov = summary.coverage * 100
    cov_style = "green" if overall_cov == 100 else "yellow" if overall_cov >= 50 else "red"

    summary_grid = Table.grid(padding=(0, 3))
    summary_grid.add_column(style="bold white")
    summary_grid.add_column()
    summary_grid.add_row("Pass Rate", f"[{cov_style}]{overall_cov:.1f}%[/{cov_style}]")
    summary_grid.add_row("Total Cases", f"{total_case_count}")
    summary_grid.add_row("Passed Cases", f"[green]{passed_case_count}[/green]")
    summary_grid.add_row(
        "Failed Cases",
        f"[bright_red]{failed_case_count}[/bright_red]" if failed_case_count else "0",
    )
    summary_grid.add_row(
        "Total Failures",
        f"[bright_red]{summary.error_count}[/bright_red]" if summary.error_count else "0",
    )

    console.print()
    console.print(Panel(summary_grid, title="Overall Summary", border_style="bright_cyan"))


def find_yaml_files(directory: Path) -> list[Path]:
    """Find all YAML files in directory recursively."""
    return sorted((*directory.glob("**/*.yml"), *directory.glob("**/*.yaml")))


def validate_coverage(ctx, param, value):
    """Validate coverage is between 0 and 100."""
    new_value = max(1, min(100, value))
    if new_value != value:
        click.secho(
            f"WARNING: Coverage expected to be in range between 1 and 100 but {int(value) if value == int(value) else value} found, defaulting to {new_value}.",
            fg="yellow",
            err=True,
        )
    return new_value


# ── Main command ─────────────────────────────────────────────────


@click.command()
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
@click.option("--verbose", "-v", is_flag=True, help="Show detailed evaluation summary")
@click.option("--hide-report", is_flag=True, help="Hide pydantic_evals report output")
def main(
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
    verbose: bool,
    hide_report: bool,
):
    """vowel — YAML-based evaluation framework for Python functions."""
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
                if not hide_report:
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
                console.print(f"\n[cyan]Changed: {Path(event.src_path).name}[/cyan]")  # type: ignore
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
        total_cases = 0
        total_passed_cases = 0
        total_errors = 0

        for yf, summary in all_summaries:
            total_errors += summary.error_count
            for r in summary.results:
                if r.report:
                    for c in r.report.cases:
                        total_cases += 1
                        if all(a.value for a in c.assertions.values()):
                            total_passed_cases += 1

            if not quiet:
                console.print()
                # Per-file table
                file_table = Table(
                    title=f"[bold white]{yf.name}[/bold white]",
                    box=box.ROUNDED,
                    show_header=True,
                    header_style="bold white",
                    expand=True,
                )
                file_table.add_column("Function", style="white bold", no_wrap=True)
                file_table.add_column("Cases", justify="center", style="white")
                file_table.add_column("Passed", justify="center", style="green")
                file_table.add_column("Failed", justify="center", style="bright_red")
                file_table.add_column("Errors", justify="center")
                file_table.add_column("Pass Rate", justify="center")

                for result in summary.results:
                    if result.error:
                        file_table.add_row(
                            result.eval_id,
                            "-",
                            "-",
                            "-",
                            "[bright_red]ERR[/bright_red]",
                            "[bright_red]—[/bright_red]",
                        )
                        continue
                    if result.report:
                        cases = result.report.cases
                        n = len(cases)
                        passed = sum(
                            1 for c in cases if all(a.value for a in c.assertions.values())
                        )
                        failed = n - passed
                        rate = passed / n * 100 if n else 100
                        rate_style = (
                            "green" if rate == 100 else "yellow" if rate >= 50 else "bright_red"
                        )
                        file_table.add_row(
                            result.eval_id,
                            str(n),
                            str(passed),
                            str(failed) if failed else "[grey50]0[/grey50]",
                            "[grey50]0[/grey50]",
                            f"[{rate_style}]{rate:.0f}%[/{rate_style}]",
                        )

                console.print(file_table)

        if not quiet:
            # Combined Overall Summary panel
            total_failed_cases = total_cases - total_passed_cases
            overall_rate = (total_passed_cases / total_cases * 100) if total_cases else 100
            rate_style = (
                "green" if overall_rate == 100 else "yellow" if overall_rate >= 50 else "bright_red"
            )

            summary_grid = Table.grid(padding=(0, 3))
            summary_grid.add_column(style="bold white")
            summary_grid.add_column()
            summary_grid.add_row("Files", str(len(all_summaries)))
            summary_grid.add_row("Total Cases", str(total_cases))
            summary_grid.add_row("Successful Evaluations", f"[green]{total_passed_cases}[/green]")
            summary_grid.add_row(
                "Failed Evaluations",
                f"[bright_red]{total_failed_cases}[/bright_red]" if total_failed_cases else "0",
            )
            summary_grid.add_row(
                "Evaluation Errors",
                f"[bright_red]{total_errors}[/bright_red]" if total_errors else "0",
            )
            summary_grid.add_row("Pass Rate", f"[{rate_style}]{overall_rate:.1f}%[/{rate_style}]")

            console.print()
            console.print(Panel(summary_grid, title="Overall Summary", border_style="bright_cyan"))

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

    if not hide_report:
        for result in summary.results:
            if result.error:
                console.print(f"[red]Error: {result.eval_id}: {result.error}[/red]")
                continue

            if result.report:
                result.report.print(include_averages=True, include_reasons=True)

    console.print()

    if verbose:
        bundle = load_bundle(yaml_files[0])
        _print_verbose_summary(console, summary, bundle, yaml_files[0])

    # Always print Overall Summary panel (verbose already includes it)
    if not verbose:
        _print_overall_summary(console, summary)

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
