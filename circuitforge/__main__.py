"""Allow `python -m circuitforge` to invoke the CircuitForge CLI."""

from circuitforge.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
