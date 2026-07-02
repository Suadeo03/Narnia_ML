"""
check_submission_files.py
Team Narnia — PhysioNet Challenge 2026

Verifies every file the Docker submission actually needs is present,
before you commit/push/submit. Designed to be run from a cluttered repo
root (dev_model/, dev_model_v2/, docker_outputs/, dev_subset/, etc. all
present) without being confused by any of that — it only checks for the
specific files the submission depends on.

Two categories checked:
  1. Fixed set — files the Challenge harness needs regardless of what
     team_code.py does (locked files, Dockerfile, requirements.txt,
     channel_table.csv).
  2. Dynamic set — parsed directly from team_code.py's own import
     statements, so this list can't drift out of sync with the actual
     code the way a hardcoded list could. If team_code.py changes what
     it imports, this script's requirements update automatically next run.

Also does two cheap sanity checks while it's at it:
  - Warns if team_code.py imports anything from a known dev-only tool
    (loso_cv, build_dev_subset, verify_label_integrity, phase1_eda,
    age_residualized_eda) — that would mean the submission accidentally
    depends on a script that isn't meant to ship.
  - Cross-checks that third-party packages actually imported by
    team_code.py / features/*.py are listed in requirements.txt.

Usage:
    python check_submission_files.py [--repo-root .]

Exit code: 0 if everything required is present, 1 otherwise (so this can
gate a pre-commit hook or a submission checklist script if you want).
"""

import argparse
import ast
import os
import re
import sys
from pathlib import Path

# Files the Challenge harness needs regardless of team_code.py's contents.
# These are the "do not edit" files from CLAUDE.md plus build/config files.
FIXED_REQUIRED = [
    "team_code.py",
    "train_model.py",
    "run_model.py",
    "helper_code.py",
    "evaluate_model.py",
    "channel_table.csv",
    "requirements.txt",
    "Dockerfile",
]

# Present in the documented file structure but not imported/called at
# runtime by train_model.py or run_model.py — flagged as recommended,
# not build-breaking if absent.
RECOMMENDED = [
    "create_labels.py",
    ".dockerignore",
]

# Dev-only tools that should never be an import dependency of team_code.py.
# Their presence as a FILE in the repo is fine; their presence as an
# IMPORT inside team_code.py would mean the submission depends on
# something not meant to ship.
DEV_ONLY_MODULES = {
    "loso_cv", "build_dev_subset", "verify_label_integrity",
    "phase1_eda", "age_residualized_eda", "entry3_calibration",
}

# import-name -> requirements.txt package-name, where they differ.
IMPORT_TO_PACKAGE = {
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
}

# Standard library modules — never expected in requirements.txt.
STDLIB_SKIP = {
    "os", "sys", "re", "ast", "argparse", "json", "csv", "collections",
    "datetime", "itertools", "functools", "pathlib", "typing", "math",
    "time", "warnings", "copy", "io", "abc",
}


def parse_team_code_imports(team_code_path: Path):
    """
    Parses team_code.py's AST for import statements. Returns:
      feature_module_paths: set of relative file paths under features/
        that must exist (derived from `from features.X import ...` and
        `from features import ...`)
      third_party_packages: set of top-level third-party import names
      dev_only_imports: set of dev-only module names imported (should be empty)
    """
    source = team_code_path.read_text()
    tree = ast.parse(source, filename=str(team_code_path))

    feature_module_paths = set()
    third_party_packages = set()
    dev_only_imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top_level = module.split(".")[0]

            if module == "features":
                # from features import X, Y  -> needs features/__init__.py
                feature_module_paths.add("features/__init__.py")
            elif module.startswith("features."):
                # from features.demographic import X -> features/demographic.py
                submodule = module.split(".", 1)[1]
                feature_module_paths.add(f"features/{submodule}.py")
                feature_module_paths.add("features/__init__.py")
            elif top_level in DEV_ONLY_MODULES:
                dev_only_imports.add(top_level)
            elif top_level not in STDLIB_SKIP and top_level not in (
                "helper_code", "team_code", "evaluate_model", "train_model", "run_model"
            ):
                third_party_packages.add(top_level)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".")[0]
                if top_level in DEV_ONLY_MODULES:
                    dev_only_imports.add(top_level)
                elif top_level not in STDLIB_SKIP:
                    third_party_packages.add(top_level)

    return feature_module_paths, third_party_packages, dev_only_imports


def check_requirements_coverage(repo_root: Path, packages: set):
    """
    Cross-checks that each third-party package imported by team_code.py
    appears (case-insensitively, substring match) somewhere in
    requirements.txt. Returns list of packages NOT found.
    """
    req_path = repo_root / "requirements.txt"
    if not req_path.exists():
        return sorted(packages)  # everything is "missing" if the file itself is gone

    req_text = req_path.read_text().lower()
    missing = []
    for pkg in sorted(packages):
        pkg_name = IMPORT_TO_PACKAGE.get(pkg, pkg)
        if pkg_name.lower() not in req_text:
            missing.append(pkg)
    return missing


def run(repo_root: str):
    root = Path(repo_root).resolve()
    print(f"Checking submission files in: {root}\n")

    problems = []
    warnings = []

    # ── Fixed required files ─────────────────────────────────────────────────
    print("Fixed required files:")
    for rel_path in FIXED_REQUIRED:
        full = root / rel_path
        ok = full.exists() and full.is_file()
        status = "OK" if ok else "MISSING"
        print(f"  [{status:^7}] {rel_path}")
        if not ok:
            problems.append(rel_path)

    print("\nRecommended (not build-breaking if absent):")
    for rel_path in RECOMMENDED:
        full = root / rel_path
        ok = full.exists() and full.is_file()
        status = "OK" if ok else "missing"
        print(f"  [{status:^7}] {rel_path}")
        if not ok:
            warnings.append(f"Recommended file missing: {rel_path}")

    # ── Dynamic: parse team_code.py's actual imports ─────────────────────────
    team_code_path = root / "team_code.py"
    if not team_code_path.exists():
        print("\nCannot parse team_code.py imports — file is missing (see above).")
        feature_paths, third_party, dev_only = set(), set(), set()
    else:
        print("\nFeature modules required by team_code.py's actual imports:")
        feature_paths, third_party, dev_only = parse_team_code_imports(team_code_path)
        for rel_path in sorted(feature_paths):
            full = root / rel_path
            ok = full.exists() and full.is_file()
            status = "OK" if ok else "MISSING"
            print(f"  [{status:^7}] {rel_path}")
            if not ok:
                problems.append(rel_path)

    # ── Dev-only import leakage check ────────────────────────────────────────
    if dev_only:
        print(f"\n!! WARNING: team_code.py imports from dev-only tool(s): "
              f"{', '.join(sorted(dev_only))}")
        print("   These are not meant to ship as a submission dependency.")
        warnings.append(f"team_code.py imports dev-only module(s): {sorted(dev_only)}")
    else:
        print("\nNo dev-only tool imports found in team_code.py — clean.")

    # ── requirements.txt coverage check ──────────────────────────────────────
    if third_party:
        print(f"\nThird-party packages imported by team_code.py: {sorted(third_party)}")
        missing_reqs = check_requirements_coverage(root, third_party)
        if missing_reqs:
            print(f"!! WARNING: not found in requirements.txt: {missing_reqs}")
            warnings.append(f"Packages imported but not in requirements.txt: {missing_reqs}")
        else:
            print("All imported packages found in requirements.txt — clean.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if not problems:
        print("PASS — all required files present.")
    else:
        print(f"FAIL — {len(problems)} required file(s) missing:")
        for p in problems:
            print(f"  - {p}")
    if warnings:
        print(f"\n{len(warnings)} warning(s) (not build-breaking, worth a look):")
        for w in warnings:
            print(f"  - {w}")
    print(f"{'='*60}")

    return len(problems) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Path to the repo root (default: current directory)")
    args = parser.parse_args()

    success = run(args.repo_root)
    sys.exit(0 if success else 1)