#!/usr/bin/env python3
"""The dimension of the RVE is decided in one place, rvegen::domain (#122).

Fails if a source line outside include/rvegen/domain.h finds the dimension
from the box's z extent again (`box[2] > 0`, `domain_box[2] == T{0}`, ...)
or picks "whichever of volume() and area() is not zero". Use
domain::is_3d() / dimension() / measure() and shape_base::measure().

A local copy (`const T lz = box[2]; if (lz > 0)`) is beyond a line-based
check; review has to catch that.

usage: no_dimension_inference_check.py <source root>
       no_dimension_inference_check.py --self-test
"""
import pathlib
import re
import sys

ROOTS = ("include", "apps", "python/src")
ALLOWED = {pathlib.PurePosixPath("include/rvegen/domain.h")}
_BOX = r"(box|dom|domain|extents|extent)\w*(\(\))?\s*\[\s*2\s*\]"
_ZERO = r"(T|value_type|double|float|static_cast|0|eps)"
_CMP = r"(>|<|>=|<=|==|!=)"
PATTERNS = (
    # box[2] > 0, _domain[2] <= value_type{0}, extents()[2] != 0.0,
    # domain_box[2] > static_cast<T>(0), box[2] > eps, ...
    re.compile(_BOX + r"\s*" + _CMP + r"\s*" + _ZERO),
    # the same with the operands swapped: T{0} < box[2], 0.0 == dom[2]
    re.compile(r"(\}|\)|\b0(\.0*)?|eps\w*)\s*" + _CMP + r"\s*!?\s*\w*" + _BOX),
    # std::get<2>(box) compared with anything
    re.compile(r"get\s*<\s*2\s*>\s*\(\s*\w*(box|dom|extent)\w*\s*\)\s*" + _CMP),
    # (v > T{0}) ? v : shape->area()  /  ? s->volume() : s->area()
)
# Choosing between volume() and area() is fine when the declared dimension
# decides (shape_base::measure()), not when a value being zero does.
TERNARY = (
    re.compile(r"\?[^:;]*:\s*[\w.>-]*(area|volume)\(\)"),
    re.compile(r"\?[^:;]*(area|volume)\(\)\s*:"),
)
DECLARED = re.compile(r"\b(dimension|is_3d|is_2d)\(\)[^?]*\?")

# Lines the patterns must catch / must leave alone (--self-test).
MUST_FLAG = (
    "if (box[2] > T{0}) {",
    "const bool is3d = domain_box[2] > 0;",
    "if (_domain_box[2] == value_type{0}) return;",
    "if (T{0} < box[2]) {",
    "if (0.0 == domain_box[2]) {",
    "if (domain_box[2] > static_cast<T>(0)) {",
    "if (domain_box[2] > eps) {",
    "if (std::get<2>(box) > 0) {",
    "auto m = (v > T{0}) ? v : shape->area();",
    "auto m = is3d ? shape->volume() : shape->area();",
)
MUST_PASS = (
    "const auto lz = domain_box[2];",
    "out[2] = box[2] * T{0.5};",
    "if (domain_box.is_3d()) {",
    "return is_3d() ? _extents[0] * _extents[1] * _extents[2] : area;",
    "sum += shape->measure();",
    "return dimension() == 3 ? volume() : area();",
    "const T v = shape->volume();",
)


def flagged(line: str) -> bool:
    code = line.split("//", 1)[0]
    if any(p.search(code) for p in PATTERNS):
        return True
    return (any(p.search(code) for p in TERNARY)
            and not DECLARED.search(code))


def self_test() -> int:
    missed = [l for l in MUST_FLAG if not flagged(l)]
    wrong = [l for l in MUST_PASS if flagged(l)]
    for l in missed:
        print(f"self-test: not flagged: {l}")
    for l in wrong:
        print(f"self-test: flagged, but fine: {l}")
    if missed or wrong:
        return 1
    print("no_dimension_inference_check self-test: OK")
    return 0


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
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
                if flagged(line):
                    hits.append(f"{rel}:{number}: {line.strip()}")
    if hits:
        print("dimension inferred outside rvegen::domain:")
        print("\n".join(hits))
        return 1
    print("no_dimension_inference_check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
