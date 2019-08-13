import argparse
import numpy as np
import pandas as pd
from synthesized.core import BasicSynthesizer
from synthesized.core.util import ProfilerArgs


def main(profile_file: str, profile_step: int, num_examples: int, num_columns: int, num_iterations: int):
    # make dummy data
    data_array = np.random.randn(num_examples, num_columns)
    column_names = ["x_{}".format(i) for i in range(num_columns)]
    indices = list(range(num_examples))
    data = pd.DataFrame(data=data_array, index=indices, columns=column_names)

    # fit synthesizer to data
    profiler_args = ProfilerArgs(filepath=profile_file, period=profile_step)
    with BasicSynthesizer(data=data, profiler_args=profiler_args) as synthesizer:
        synthesizer.learn(data=data, num_iterations=num_iterations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--profile-file', type=str, help="profile log file")
    parser.add_argument('-s', '--profile-step', type=int, help="profile trace frequency")
    parser.add_argument('-e', '--num-examples', type=int, help="number of examples")
    parser.add_argument('-c', '--num-columns', type=int, help="number of columns")
    parser.add_argument('-i', '--num-iterations', type=int, help="number of training iterations")
    args = parser.parse_args()
    main(profile_file=args.profile_file, profile_step=args.profile_step,
         num_examples=args.num_examples, num_columns=args.num_columns,
         num_iterations=args.num_iterations)
