#!/usr/bin/env python3
"""No standard-library random transform in the library (#130).

The engines of <random> are fully specified; the distributions and the
shuffling algorithms are not, and libstdc++, libc++ and MSVC turn the same
engine output into different values. rvegen samples through
rvegen/distributions/portable_random.h instead, and this check fails on a
source line that uses a std:: distribution, std::generate_canonical,
std::shuffle, std::sample or std::random_device again.

usage: no_std_distribution_check.py <source root>
"""
import pathlib
import re
import sys

ROOTS = ("include", "apps", "python/src")
# `using namespace std;` would let a bare normal_distribution through (and
# rvegen has classes of its own with these names), so it is refused too.
PATTERN = re.compile(
    r"std::(\w+_distribution|generate_canonical|shuffle|random_shuffle|sample"
    r"|random_device)\b|\busing\s+namespace\s+std\s*;")


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    hits = []
    for sub in ROOTS:
        if not (root / sub).is_dir():
            print(f"no_std_distribution_check: no directory {root / sub}")
            return 2
        for path in sorted((root / sub).rglob("*")):
            if path.suffix not in (".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx"):
                continue
            rel = path.relative_to(root).as_posix()
            for number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), 1):
                code = line.split("//", 1)[0]
                if PATTERN.search(code):
                    hits.append(f"{rel}:{number}: {line.strip()}")
    if hits:
        print("standard-library random transform (use portable_random.h):")
        print("\n".join(hits))
        return 1
    print("no_std_distribution_check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
