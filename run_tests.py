"""One-command regression: published anchors + stochastic smoke tests.

Run:  python run_tests.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    suites = [
        HERE / "tests" / "test_anchors.py",
        HERE / "tests" / "test_sim.py",
        HERE / "tests" / "test_validation.py",
    ]
    failed = []
    for suite in suites:
        print(f"\n>>> {suite.relative_to(HERE)}")
        rc = subprocess.call([sys.executable, str(suite)])
        if rc != 0:
            failed.append(suite.name)
    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED suites: {', '.join(failed)}")
        return 1
    print("ALL TEST SUITES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
