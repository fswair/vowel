"""Tests for CLI behavior outside watch mode."""

import json

from click.testing import CliRunner

from vowel.cli import main


class TestCliExportJson:
    """Test JSON export behavior."""

    def test_export_json_writes_object_payload(self, tmp_path):
        """--export-json should write a JSON object, not a quoted string."""
        yaml_path = tmp_path / "evals.yml"
        export_path = tmp_path / "results.json"
        yaml_path.write_text(
            """
len:
  dataset:
    - case:
        input: [1, 2, 3]
        expected: 3
"""
        )

        runner = CliRunner()
        result = runner.invoke(main, [str(yaml_path), "--export-json", str(export_path), "--quiet"])

        assert result.exit_code == 0

        payload = json.loads(export_path.read_text())
        assert isinstance(payload, dict)
        assert "summary" in payload
        assert "results" in payload
