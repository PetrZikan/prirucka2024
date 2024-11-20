from click.testing import CliRunner
from prirucka2024.__main__ import main


def test_cli_runs():
    """Test if the CLI initializes without errors."""
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0
    assert "Usage" in result.output
