"""Tests for CLI watch mode."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from vowel.cli import main


class TestWatchMode:
    """Test --watch CLI option."""

    def test_watch_option_exists(self):
        """Test that --watch option is available."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "-w, --watch" in result.output
        assert "Watch mode" in result.output

    def test_watch_requires_yaml_file(self):
        """Test that --watch requires a YAML file argument."""
        runner = CliRunner()
        result = runner.invoke(main, ["--watch"])
        assert result.exit_code != 0
        assert "requires a YAML file" in result.output or "Missing argument" in result.output

    def test_watch_with_nonexistent_file(self):
        """Test --watch with non-existent file."""
        runner = CliRunner()
        result = runner.invoke(main, ["nonexistent.yml", "--watch"])
        assert result.exit_code != 0

    @patch("watchdog.observers.Observer")
    @patch("vowel.cli.run_evals")
    def test_watch_starts_observer(self, mock_run_evals, mock_observer_class):
        """Test that --watch starts the file observer."""
        import os
        import tempfile

        # Create a temp yaml file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("test_func:\n  dataset:\n    - case: {id: test, input: 1, expected: 1}\n")
            temp_path = f.name

        try:
            # Mock the observer
            mock_observer = MagicMock()
            mock_observer_class.return_value = mock_observer

            # Mock run_evals to return a summary
            mock_summary = MagicMock()
            mock_summary.all_passed = True
            mock_summary.total_count = 1
            mock_summary.results = []
            mock_run_evals.return_value = mock_summary

            runner = CliRunner()

            # This will timeout/interrupt, so we catch it
            with patch("time.sleep", side_effect=KeyboardInterrupt):
                runner.invoke(main, [temp_path, "--watch"])

            # Observer should have been started
            mock_observer.start.assert_called_once()
        finally:
            os.unlink(temp_path)
