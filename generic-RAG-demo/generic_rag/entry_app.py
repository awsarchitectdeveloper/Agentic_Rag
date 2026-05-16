#!/usr/bin/env python3
from pathlib import Path
from chainlit.cli import run_chainlit

PACKAGE_ROOT = Path(__file__).resolve().parent


def main():
    run_chainlit(str(PACKAGE_ROOT / "app.py"))


if __name__ == "__main__":
    main()
