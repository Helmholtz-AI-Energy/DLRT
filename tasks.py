"""
Tasks for maintaining the project.

Execute 'invoke --list' for guidance on using Invoke
"""
from __future__ import annotations

import platform
import webbrowser
from pathlib import Path

from invoke import call
from invoke import task
from invoke.context import Context
from invoke.runners import Result

ROOT_DIR = Path(__file__).parent
DOCS_DIR = ROOT_DIR.joinpath("docs")
DOCS_BUILD_DIR = DOCS_DIR.joinpath("_build")
DOCS_INDEX = DOCS_BUILD_DIR.joinpath("index.html")
COVERAGE_FILE = ROOT_DIR.joinpath(".coverage")
COVERAGE_DIR = ROOT_DIR.joinpath("htmlcov")
COVERAGE_REPORT = COVERAGE_DIR.joinpath("index.html")
SOURCE_DIR = ROOT_DIR.joinpath("src/dlrt")
TEST_DIR = ROOT_DIR.joinpath("tests")
PYTHON_TARGETS = [
    SOURCE_DIR,
    TEST_DIR,
    Path(__file__),
]
PYTHON_TARGETS_STR = " ".join([str(p) for p in PYTHON_TARGETS])


def _run(c: Context, command: str) -> Result:
    return c.run(command, pty=platform.system() != "Windows")


@task()
def clean_build(c):
    # type: (Context) -> None
    """Clean up files from package building."""
    _run(c, "rm -fr build/")
    _run(c, "rm -fr dist/")
    _run(c, "rm -fr .eggs/")
    _run(c, "find . -name '*.egg-info' -exec rm -fr {} +")
    _run(c, "find . -name '*.egg' -exec rm -f {} +")


@task()
def clean_python(c):
    # type: (Context) -> None
    """Clean up python file artifacts."""
    _run(c, "find . -name '*.pyc' -exec rm -f {} +")
    _run(c, "find . -name '*.pyo' -exec rm -f {} +")
    _run(c, "find . -name '*~' -exec rm -f {} +")
    _run(c, "find . -name '__pycache__' -exec rm -fr {} +")


@task()
def clean_tests(c):
    # type: (Context) -> None
    """Clean up files from testing."""
    _run(c, f"rm -f {COVERAGE_FILE}")
    _run(c, f"rm -fr {COVERAGE_DIR}")
    _run(c, "rm -fr .pytest_cache")


@task()
def clean_docs(c):
    # type: (Context) -> None
    """Clean up files from documentation builds."""
    _run(c, f"rm -fr {DOCS_BUILD_DIR}")


@task(pre=[clean_build, clean_python, clean_tests, clean_docs])
def clean(c):
    # type: (Context) -> None
    """Run all clean sub-tasks."""


@task()
def install_hooks(c):
    # type: (Context) -> None
    """Install pre-commit hooks."""
    _run(c, "pre-commit install")


@task()
def hooks(c):
    # type: (Context) -> None
    """Run pre-commit hooks."""
    _run(c, "pre-commit run --all")
