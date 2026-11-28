#!/usr/bin/env python3
"""The dimension of the RVE is decided in one place, rvegen::domain (#122).

Fails if a source line outside include/rvegen/domain.h finds the dimension
from the box's z extent again (`box[2] > 0`, `domain_box[2] == T{0}`, ...)
or picks "whichever of volume() and area() is not zero". Use
domain::is_3d() / dimension() / measure() and shape_base::measure().

usage: no_dimension_inference_check.py <source root>
"""
import pathlib
import re
import sys

ROOTS = ("include", "apps", "python/src")
ALLOWED = {pathlib.PurePosixPath("include/rvegen/domain.h")}
PATTERNS = (
    # box[2] > 0, _domain[2] <= value_type{0}, extents()[2] != 0.0, ...
    re.compile(r"(box|dom|domain|extents|extent)\w*(\(\))?\s*\[\s*2\s*\]\s*"
               r"(>|<|>=|<=|==|!=)\s*(T|value_type|double|float|0)"),
    # (v > T{0}) ? v : shape->area()
    re.compile(r"\?\s*\w+\s*:\s*[\w.>-]*area\(\)"),
    re.compile(r"\?\s*\w+\s*:\s*[\w.>-]*volume\(\)"),
)


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    hits = []
    for sub in ROOTS:
        for path in sorted((root / sub).rglob("*")):
            if path.suffix not in (".h", ".hpp", ".cpp"):
                continue
            rel = pathlib.PurePosixPath(path.relative_to(root).as_posix())
            if rel in ALLOWED:
                continue
            for number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), 1):
                code = line.split("//", 1)[0]
                if any(p.search(code) for p in PATTERNS):
                    hits.append(f"{rel}:{number}: {line.strip()}")
    if hits:
        print("dimension inferred outside rvegen::domain:")
        print("\n".join(hits))
        return 1
    print("no_dimension_inference_check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
