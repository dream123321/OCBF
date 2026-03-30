import sys

from ocbf.cli import main


if __name__ == "__main__":
    argv = sys.argv[1:] or ["run", "ocbf.ocbf.per_configuration.example.json"]
    raise SystemExit(main(argv))
