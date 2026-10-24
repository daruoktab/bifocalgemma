"""Compare one completed BifocalGemma benchmark directory with its matching IT baseline."""

import argparse
import json

from bifocalgemma.evaluation.comparison import compare_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_directory")
    parser.add_argument("baseline_directory")
    args = parser.parse_args()
    print(json.dumps(compare_predictions(args.model_directory, args.baseline_directory), indent=2))


if __name__ == "__main__":
    main()
