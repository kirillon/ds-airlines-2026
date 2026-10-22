import argparse
from math import inf
from pathlib import Path

from parsers.xlsx import parse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Input file (zip file with xls)")
    parser.add_argument("output", type=Path, help="Output file (parquet)")

    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("Input file must exist")
    df = parse.parse_zip_xlsx(args.input)
    df.to_parquet(args.output)


if __name__ == "__main__":
    main()
