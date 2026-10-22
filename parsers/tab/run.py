import argparse
from pathlib import Path

from parsers.tab import parse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Input file (.tab)")
    parser.add_argument("output", type=Path, help="Output file")

    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("Input file must exist")
    with args.input.open(encoding="utf-8") as infile:
        indata = infile.read()
    df = parse.parse_tab(indata)
    df.to_parquet(args.output)


if __name__ == "__main__":
    main()
