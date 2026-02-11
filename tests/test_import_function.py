"""Tests for import_function utility."""

import os
import sys

import pytest

from vowel.utils import import_function


class TestImportBuiltins:
    """Tests for importing builtin functions."""

    def test_import_len(self):
        """Test importing builtin len."""
        func = import_function("len")

        assert func is len
        assert func([1, 2, 3]) == 3

    def test_import_abs(self):
        """Test importing builtin abs."""
        func = import_function("abs")

        assert func is abs
        assert func(-5) == 5

    def test_import_sum(self):
        """Test importing builtin sum."""
        func = import_function("sum")

        assert func is sum
        assert func([1, 2, 3]) == 6

    def test_import_max(self):
        """Test importing builtin max."""
        func = import_function("max")

        assert func is max
        assert func([1, 5, 3]) == 5

    def test_import_min(self):
        """Test importing builtin min."""
        func = import_function("min")

        assert func is min
        assert func([1, 5, 3]) == 1

    def test_import_sorted(self):
        """Test importing builtin sorted."""
        func = import_function("sorted")

        assert func is sorted
        assert func([3, 1, 2]) == [1, 2, 3]

    def test_import_nonexistent_builtin_raises(self):
        """Test importing nonexistent builtin raises error."""
        with pytest.raises(ImportError):
            import_function("nonexistent_builtin")


class TestImportStdlib:
    """Tests for importing stdlib functions."""

    def test_import_math_sqrt(self):
        """Test importing math.sqrt."""
        func = import_function("math.sqrt")

        assert func(16) == 4.0

    def test_import_math_floor(self):
        """Test importing math.floor."""
        func = import_function("math.floor")

        assert func(3.7) == 3

    def test_import_math_ceil(self):
        """Test importing math.ceil."""
        func = import_function("math.ceil")

        assert func(3.2) == 4

    def test_import_os_path_join(self):
        """Test importing os.path.join."""
        func = import_function("os.path.join")

        result = func("a", "b", "c")
        assert "a" in result and "b" in result and "c" in result

    def test_import_os_path_basename(self):
        """Test importing os.path.basename."""
        func = import_function("os.path.basename")

        assert func("/path/to/file.txt") == "file.txt"

    def test_import_json_dumps(self):
        """Test importing json.dumps."""
        func = import_function("json.dumps")

        result = func({"a": 1})
        assert result == '{"a": 1}'

    def test_import_json_loads(self):
        """Test importing json.loads."""
        func = import_function("json.loads")

        result = func('{"a": 1}')
        assert result == {"a": 1}

    def test_import_re_match(self):
        """Test importing re.match."""
        func = import_function("re.match")

        result = func(r"\d+", "123abc")
        assert result is not None

    def test_import_collections_defaultdict(self):
        """Test importing collections.defaultdict."""
        cls = import_function("collections.defaultdict")

        d = cls(int)
        d["key"] += 1
        assert d["key"] == 1


class TestImportBuiltinMethods:
    """Tests for importing builtin type methods."""

    def test_import_str_upper(self):
        """Test importing str.upper."""
        func = import_function("str.upper")

        assert func("hello") == "HELLO"

    def test_import_str_lower(self):
        """Test importing str.lower."""
        func = import_function("str.lower")

        assert func("HELLO") == "hello"

    def test_import_str_strip(self):
        """Test importing str.strip."""
        func = import_function("str.strip")

        assert func("  hello  ") == "hello"

    def test_import_list_sort(self):
        """Test importing list.sort."""
        func = import_function("list.sort")

        items = [3, 1, 2]
        func(items)
        assert items == [1, 2, 3]


class TestImportLocalModules:
    """Tests for importing from local modules."""

    def test_import_local_function(self, tmp_path):
        """Test importing function from local module."""
        module_file = tmp_path / "my_module.py"
        module_file.write_text(
            """
def my_function(x):
    return x * 2
"""
        )

        original_path = sys.path.copy()
        original_cwd = os.getcwd()

        try:
            sys.path.insert(0, str(tmp_path))
            os.chdir(tmp_path)

            func = import_function("my_module.my_function")

            assert func(5) == 10
        finally:
            sys.path = original_path
            os.chdir(original_cwd)

    def test_import_nested_module(self, tmp_path):
        """Test importing from nested module structure."""
        pkg_dir = tmp_path / "mypkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "utils.py").write_text(
            """
def helper(x):
    return x + 1
"""
        )

        original_path = sys.path.copy()
        original_cwd = os.getcwd()

        try:
            sys.path.insert(0, str(tmp_path))
            os.chdir(tmp_path)

            func = import_function("mypkg.utils.helper")

            assert func(5) == 6
        finally:
            sys.path = original_path
            os.chdir(original_cwd)


class TestImportErrors:
    """Tests for import error handling."""

    def test_import_nonexistent_module(self):
        """Test importing from nonexistent module."""
        with pytest.raises(ImportError):
            import_function("nonexistent_module.function")

    def test_import_nonexistent_function(self):
        """Test importing nonexistent function from valid module."""
        with pytest.raises(ImportError):
            import_function("math.nonexistent_function")

    def test_import_invalid_path(self):
        """Test importing with invalid path format."""
        with pytest.raises(ImportError):
            import_function("not_a_builtin")


class TestImportWithRunEvals:
    """Integration tests for import_function with RunEvals."""

    def test_run_evals_auto_imports_math(self):
        """Test RunEvals auto-imports math functions."""
        spec = {
            "math.sqrt": {
                "dataset": [
                    {"case": {"input": 16, "expected": 4.0}},
                    {"case": {"input": 9, "expected": 3.0}},
                ]
            }
        }

        from vowel import RunEvals

        summary = RunEvals.from_dict(spec).run()

        assert summary.all_passed

    def test_run_evals_auto_imports_builtins(self):
        """Test RunEvals auto-imports builtin functions."""
        spec = {
            "len": {
                "dataset": [
                    {"case": {"input": [1, 2, 3], "expected": 3}},
                ]
            },
            "abs": {
                "dataset": [
                    {"case": {"input": -5, "expected": 5}},
                ]
            },
        }

        from vowel import RunEvals

        summary = RunEvals.from_dict(spec).run()

        assert summary.all_passed

    def test_run_evals_auto_imports_os_path(self):
        """Test RunEvals auto-imports os.path functions."""
        spec = {
            "os.path.basename": {
                "dataset": [
                    {"case": {"input": "/path/to/file.txt", "expected": "file.txt"}},
                ]
            }
        }

        from vowel import RunEvals

        summary = RunEvals.from_dict(spec).run()

        assert summary.all_passed
