#!/usr/bin/env python3
"""Fail if the type catalogue embedded in README.md drifted from the registries.

README.md carries the output of `rvegen --list-types` between the markers
<!-- BEGIN list-types --> and <!-- END list-types -->. To refresh it:

    ./build/rvegen --list-types   # paste between the markers
    # or: tests/readme_types_check.py --rvegen build/rvegen --readme README.md --update
"""
import argparse
import subprocess
import sys
from pathlib import Path

BEGIN, END = "<!-- BEGIN list-types -->", "<!-- END list-types -->"

ap = argparse.ArgumentParser()
ap.add_argument("--rvegen", required=True)
ap.add_argument("--readme", required=True)
ap.add_argument("--update", action="store_true")
a = ap.parse_args()

listing = subprocess.run([a.rvegen, "--list-types"], capture_output=True,
                         text=True, check=True).stdout.strip()
readme = Path(a.readme)
text = readme.read_text()
if BEGIN not in text or END not in text:
    sys.exit(f"{readme}: missing {BEGIN} / {END} markers")
head, rest = text.split(BEGIN, 1)
body, tail = rest.split(END, 1)
if body.strip() == listing:
    print("README type catalogue is up to date")
    sys.exit(0)
if a.update:
    readme.write_text(f"{head}{BEGIN}\n{listing}\n{END}{tail}")
    print("README updated")
    sys.exit(0)
print("README type catalogue is stale. Registered types are:\n" + listing)
print("\nRun tests/readme_types_check.py --update to refresh it.")
sys.exit(1)
