#!/usr/bin/env python3
"""
Minimal dependency-free test runner.

Prefer `pytest tests/` if pytest is available in your environment; this
script exists only as a fallback that discovers and runs every function
named `test_*` in tests/test_*.py using plain `assert` statements, with no
extra dependency beyond the project's own requirements.
"""

import importlib
import sys
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).parent / "tests"


def discover_test_modules():
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        module_name = f"tests.{path.stem}"
        yield module_name


def main() -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    total, failed = 0, 0

    for module_name in discover_test_modules():
        module = importlib.import_module(module_name)
        test_funcs = [
            getattr(module, name)
            for name in dir(module)
            if name.startswith("test_") and callable(getattr(module, name))
        ]
        for func in test_funcs:
            total += 1
            try:
                func()
                print(f"PASS  {module_name}.{func.__name__}")
            except Exception:
                failed += 1
                print(f"FAIL  {module_name}.{func.__name__}")
                traceback.print_exc()

    print(f"\n{total - failed}/{total} tests passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
