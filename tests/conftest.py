import pytest


@pytest.fixture(autouse=True)
def isolate_arenapilot_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARENAPILOT_HOME", str(tmp_path / ".arenapilot-home"))
