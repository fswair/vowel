"""Tests for the fixture mechanism."""

import pytest

from vowel.eval_types import FixtureDefinition
from vowel.runner import RunEvals
from vowel.utils import (
    EvalsBundle,
    FixtureManager,
    FixtureSignatureError,
    load_bundle,
    validate_fixture_signature,
)


def valid_with_fixture(name: str, age: int, *, db: object) -> str:
    """Valid: fixture is keyword-only at the end."""
    return f"{name} ({age}) - {db}"


def valid_with_multiple_fixtures(x: int, *, db: object, cache: object) -> int:
    """Valid: multiple fixtures, all keyword-only."""
    return x


def valid_no_fixtures(name: str, age: int) -> str:
    """Valid: no fixtures at all."""
    return f"{name} ({age})"


def invalid_fixture_positional(db: object, name: str) -> str:
    """Invalid: fixture is positional (at start)."""
    return f"{name} - {db}"


def invalid_fixture_not_kwonly(name: str, db: object) -> str:
    """Invalid: fixture is not keyword-only."""
    return f"{name} - {db}"


def invalid_with_args(*args, db: object) -> str:
    """Invalid: has *args."""
    return str(args)


def invalid_with_kwargs(name: str, **kwargs) -> str:
    """Invalid: has **kwargs."""
    return name


def invalid_with_both(*args, **kwargs) -> str:
    """Invalid: has both *args and **kwargs."""
    return ""


class TestValidateFixtureSignature:
    def test_valid_single_fixture(self):
        """Should pass with valid keyword-only fixture."""
        validate_fixture_signature(valid_with_fixture, ["db"])

    def test_valid_multiple_fixtures(self):
        """Should pass with multiple keyword-only fixtures."""
        validate_fixture_signature(valid_with_multiple_fixtures, ["db", "cache"])

    def test_valid_no_fixtures(self):
        """Should pass when no fixtures are requested."""
        validate_fixture_signature(valid_no_fixtures, [])

    def test_valid_empty_fixture_list(self):
        """Should pass when fixture list is empty."""
        validate_fixture_signature(valid_with_fixture, [])

    def test_invalid_fixture_positional(self):
        """Should fail when fixture is positional."""
        with pytest.raises(FixtureSignatureError) as exc:
            validate_fixture_signature(invalid_fixture_positional, ["db"])
        assert "keyword-only" in str(exc.value)

    def test_invalid_fixture_not_kwonly(self):
        """Should fail when fixture is not keyword-only."""
        with pytest.raises(FixtureSignatureError) as exc:
            validate_fixture_signature(invalid_fixture_not_kwonly, ["db"])
        assert "keyword-only" in str(exc.value)

    def test_invalid_with_args(self):
        """Should fail when function has *args."""
        with pytest.raises(FixtureSignatureError) as exc:
            validate_fixture_signature(invalid_with_args, ["db"])
        assert "*args" in str(exc.value)

    def test_invalid_with_kwargs(self):
        """Should fail when function has **kwargs."""
        with pytest.raises(FixtureSignatureError) as exc:
            validate_fixture_signature(invalid_with_kwargs, ["db"])
        assert "**kwargs" in str(exc.value)

    def test_invalid_with_both(self):
        """Should fail when function has both *args and **kwargs."""
        with pytest.raises(FixtureSignatureError) as exc:
            validate_fixture_signature(invalid_with_both, ["db"])
        assert "*args" in str(exc.value)

    def test_missing_fixture_param(self):
        """Should fail when fixture is not in function params."""
        with pytest.raises(FixtureSignatureError) as exc:
            validate_fixture_signature(valid_no_fixtures, ["db"])
        assert "doesn't have a parameter" in str(exc.value)


_db_instances = []
_cache_instances = []


def setup_db(host: str = "localhost", port: int = 5432):
    """Simulated DB setup."""
    instance = {"type": "db", "host": host, "port": port, "connected": True}
    _db_instances.append(instance)
    return instance


def teardown_db(instance):
    """Simulated DB teardown."""
    instance["connected"] = False
    _db_instances.remove(instance)


def setup_cache():
    """Simulated cache setup."""
    instance = {"type": "cache", "items": {}}
    _cache_instances.append(instance)
    return instance


def teardown_cache(instance):
    """Simulated cache teardown."""
    _cache_instances.remove(instance)


class TestFixtureManager:
    def setup_method(self):
        """Clear instances before each test."""
        _db_instances.clear()
        _cache_instances.clear()

    def test_setup_function_scope(self):
        """Should setup function-scoped fixture."""
        fixtures = {
            "db": FixtureDefinition(
                setup="test_fixtures.setup_db",
                teardown="test_fixtures.teardown_db",
                scope="function",
                params={"host": "test-host", "port": 5555},
            )
        }
        manager = FixtureManager(fixtures)

        instance = manager.setup("db")

        assert instance["type"] == "db"
        assert instance["host"] == "test-host"
        assert instance["port"] == 5555
        assert instance["connected"] is True
        assert len(_db_instances) == 1

    def test_teardown_function_scope(self):
        """Should teardown function-scoped fixture."""
        fixtures = {
            "db": FixtureDefinition(
                setup="test_fixtures.setup_db",
                teardown="test_fixtures.teardown_db",
                scope="function",
            )
        }
        manager = FixtureManager(fixtures)

        manager.setup("db")
        assert len(_db_instances) == 1

        manager.teardown("db", "function")
        assert len(_db_instances) == 0

    def test_module_scope_caching(self):
        """Should cache module-scoped fixtures."""
        fixtures = {
            "db": FixtureDefinition(
                setup="test_fixtures.setup_db",
                teardown="test_fixtures.teardown_db",
                scope="module",
            )
        }
        manager = FixtureManager(fixtures)

        instance1 = manager.setup("db")
        instance2 = manager.setup("db")

        assert instance1 is instance2
        assert len(_db_instances) == 1

    def test_session_scope_caching(self):
        """Should cache session-scoped fixtures."""
        fixtures = {
            "cache": FixtureDefinition(
                setup="test_fixtures.setup_cache",
                teardown="test_fixtures.teardown_cache",
                scope="session",
            )
        }
        manager = FixtureManager(fixtures)

        instance1 = manager.setup("cache")
        instance2 = manager.setup("cache")

        assert instance1 is instance2
        assert len(_cache_instances) == 1

    def test_get_fixtures_for_function(self):
        """Should setup and return all fixtures for a function."""
        fixtures = {
            "db": FixtureDefinition(
                setup="test_fixtures.setup_db",
                scope="function",
            ),
            "cache": FixtureDefinition(
                setup="test_fixtures.setup_cache",
                scope="function",
            ),
        }
        manager = FixtureManager(fixtures)

        result = manager.get_fixtures_for_function(["db", "cache"])

        assert "db" in result
        assert "cache" in result
        assert result["db"]["type"] == "db"
        assert result["cache"]["type"] == "cache"

    def test_teardown_all(self):
        """Should teardown all fixtures of a given scope."""
        fixtures = {
            "db": FixtureDefinition(
                setup="test_fixtures.setup_db",
                teardown="test_fixtures.teardown_db",
                scope="session",
            ),
            "cache": FixtureDefinition(
                setup="test_fixtures.setup_cache",
                teardown="test_fixtures.teardown_cache",
                scope="session",
            ),
        }
        manager = FixtureManager(fixtures)

        manager.setup("db")
        manager.setup("cache")
        assert len(_db_instances) == 1
        assert len(_cache_instances) == 1

        manager.teardown_all("session")
        assert len(_db_instances) == 0
        assert len(_cache_instances) == 0

    def test_unknown_fixture_error(self):
        """Should raise error for unknown fixture."""
        fixtures = {}
        manager = FixtureManager(fixtures)

        with pytest.raises(ValueError) as exc:
            manager.setup("unknown")
        assert "Unknown fixture" in str(exc.value)


class TestBundleLoading:
    def test_load_bundle_with_fixtures(self):
        """Should load both evals and fixtures from YAML."""
        yaml_content = """
fixtures:
  db:
    setup: test_fixtures.setup_db
    teardown: test_fixtures.teardown_db
    scope: module
    params:
      host: 192.168.1.1
      port: 5432

examples.functions.add_numbers:
  fixture:
    - db
  evals:
    check_result:
      equals: true
  dataset:
    - case:
        input: {a: 1, b: 2}
        expected: 3
"""
        bundle = load_bundle(yaml_content)

        assert isinstance(bundle, EvalsBundle)
        assert "db" in bundle.fixtures
        assert bundle.fixtures["db"].setup == "test_fixtures.setup_db"
        assert bundle.fixtures["db"].scope == "module"
        assert bundle.fixtures["db"].params == {"host": "192.168.1.1", "port": 5432}

    def test_load_bundle_without_fixtures(self):
        """Should load evals without fixtures."""
        yaml_content = """
examples.functions.add_numbers:
  evals:
    check_result:
      equals: true
  dataset:
    - case:
        input: {a: 1, b: 2}
        expected: 3
"""
        bundle = load_bundle(yaml_content)

        assert isinstance(bundle, EvalsBundle)
        assert bundle.fixtures == {}
        assert "examples.functions.add_numbers" in bundle.evals

    def test_eval_fixture_field(self):
        """Should parse fixture field in eval."""
        yaml_content = """
fixtures:
  db:
    setup: test_fixtures.setup_db

examples.functions.add_numbers:
  fixture:
    - db
  evals:
    check_result:
      equals: true
  dataset:
    - case:
        input: {a: 1, b: 2}
        expected: 3
"""
        bundle = load_bundle(yaml_content)
        evals = bundle.evals["examples.functions.add_numbers"]

        assert evals.fixture == ["db"]


def function_with_db(a: int, b: int, *, db: dict) -> int:
    """Test function that uses a db fixture."""
    if db and db.get("connected"):
        return a + b + db.get("port", 0)
    return a + b


class TestIntegration:
    def setup_method(self):
        _db_instances.clear()
        _cache_instances.clear()

    def test_fixture_injection_valid_signature(self):
        """Should validate and use fixtures correctly."""
        fixtures = {
            "db": FixtureDefinition(
                setup="test_fixtures.setup_db",
                teardown="test_fixtures.teardown_db",
                scope="function",
                params={"port": 100},
            )
        }

        validate_fixture_signature(function_with_db, ["db"])

        manager = FixtureManager(fixtures)
        db_instance = manager.setup("db")

        result = function_with_db(1, 2, db=db_instance)

        assert result == 103

        manager.teardown("db", "function")
        assert len(_db_instances) == 0


def add_with_db(a: int, b: int, *, db: dict) -> int:
    """Function that uses db fixture."""
    return a + b + db.get("bonus", 0)


class TestProgrammaticFixtures:
    def setup_method(self):
        _db_instances.clear()
        _cache_instances.clear()

    def test_with_fixtures_setup_only(self):
        """Should work with setup-only fixtures via with_fixtures."""
        yaml_content = """
add_with_db:
  fixture:
    - db
  dataset:
    - case:
        inputs: {a: 1, b: 2}
        expected: 13
"""

        def create_db():
            return {"bonus": 10, "connected": True}

        summary = (
            RunEvals.from_source(yaml_content)
            .with_functions({"add_with_db": add_with_db})
            .with_fixtures({"db": create_db})
            .run()
        )

        assert summary.all_passed

    def test_with_fixtures_with_teardown(self):
        """Should work with setup and teardown via with_fixtures."""
        yaml_content = """
add_with_db:
  fixture:
    - db
  dataset:
    - case:
        inputs: {a: 1, b: 2}
        expected: 103
"""
        teardown_called = []

        def create_db():
            instance = {"bonus": 100, "connected": True}
            _db_instances.append(instance)
            return instance

        def close_db(db):
            teardown_called.append(db)
            _db_instances.remove(db)

        summary = (
            RunEvals.from_source(yaml_content)
            .with_functions({"add_with_db": add_with_db})
            .with_fixtures({"db": (create_db, close_db)})
            .run()
        )

        assert summary.all_passed
        assert len(teardown_called) == 1
        assert len(_db_instances) == 0

    def test_with_fixtures_multiple(self):
        """Should work with multiple fixtures."""
        yaml_content = """
multi_fixture_func:
  fixture:
    - db
    - cache
  dataset:
    - case:
        inputs: {x: 5}
        expected: 15
"""

        def create_db():
            return {"bonus": 5}

        def create_cache():
            return {"multiplier": 2}

        def multi_fixture_func(x: int, *, db: dict, cache: dict) -> int:
            return x * cache["multiplier"] + db["bonus"]

        summary = (
            RunEvals.from_source(yaml_content)
            .with_functions({"multi_fixture_func": multi_fixture_func})
            .with_fixtures(
                {
                    "db": create_db,
                    "cache": create_cache,
                }
            )
            .run()
        )

        assert summary.all_passed

    def test_with_fixtures_chaining(self):
        """Should allow chaining with_fixtures calls."""
        yaml_content = """
add_with_db:
  fixture:
    - db
  dataset:
    - case:
        inputs: {a: 1, b: 2}
        expected: 8
"""

        summary = (
            RunEvals.from_source(yaml_content)
            .with_functions({"add_with_db": add_with_db})
            .with_fixtures({"db": lambda: {"bonus": 5}})
            .run()
        )

        assert summary.all_passed

    def test_fixture_missing_error(self):
        """Should raise error when fixture is not provided."""
        yaml_content = """
add_with_db:
  fixture:
    - db
  dataset:
    - case:
        inputs: {a: 1, b: 2}
        expected: 3
"""

        summary = (
            RunEvals.from_source(yaml_content).with_functions({"add_with_db": add_with_db}).run()
        )

        assert not summary.all_passed
        assert summary.error_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
