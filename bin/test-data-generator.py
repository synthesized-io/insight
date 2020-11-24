#!/usr/bin/env python

import argparse
import logging

from synthesized.complex import TestingDataGenerator

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Testing Data Generator.")
    parser.add_argument("--config", dest="config", required=True,
                        help="YAML configuration file.")
    parser.add_argument("--output-file", dest="output_file", required=True,
                        help="Output file.")
    parser.add_argument("-n", "--num-rows", dest="num_rows", required=True, type=int,
                        help="Number of rows to generate.")
    parser.add_argument("--output-format", dest="output_format", required=False,
                        default="csv", help="")
    args = parser.parse_args()

    generator = TestingDataGenerator.from_yaml(args.config)
    df_synthesized = generator.synthesize(args.num_rows)

    if args.output_format == "csv":
        df_synthesized.to_csv(args.output_file, index=False)
    elif args.output_format == "json":
        df_synthesized.to_json(args.output_file, orient='records', indent=2)


if __name__ == '__main__':
    main()
