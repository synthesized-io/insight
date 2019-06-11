import tqdm
import argh
import itertools
from synthesized.testing.synthetic_distributions import *
from synthesized.core.util import ProfilerArgs
from memory_profiler import memory_usage


def create_multidimensional_gaussian(dimensions: int, size: int) -> pd.DataFrame:
    """Draw `size` samples from a `dimensions`-dimensional standard gaussian. """
    z = np.random.randn(size, dimensions)
    columns = ["x_{}".format(i) for i in range(dimensions)]
    df = pd.DataFrame(z, columns=columns)
    return df


def create_multidimensional_categorical(dimensions: int, categories: int, size: int) -> pd.DataFrame:
    """Draw `size` samples from a `dimensions`-dimensional standard gaussian.  """
    z = np.random.choice(list(range(categories)), dimensions*size).reshape(size, dimensions)
    columns = ["x_{}".format(i) for i in range(dimensions)]
    df = pd.DataFrame(z, columns=columns)
    return df


def time_synthesis(data: pd.DataFrame, num_iterations: int) -> Dict[str, float]:
    t0 = time.time()
    with BasicSynthesizer(data=data) as synthesizer:
        synthesizer.learn(data=data, num_iterations=num_iterations)
        t1 = time.time()
        synthesized = synthesizer.synthesize(n=len(data))
        t2 = time.time()
        distances = [ks_2samp(data[col], synthesized[col])[0] for col in data.columns]
        avg_distance = np.mean(distances)
    return {"learn_time": t1 - t0, "synthesis_time": t2 - t1, "mean_ks": avg_distance}


def generate_argument_combinations(arg_dict):
    args_tuples = list(arg_dict.items())
    keys, values = zip(*args_tuples)
    combinations = list(itertools.product(*values))
    args_list = [dict(zip(keys, value)) for value in combinations]
    return args_list


def run_profiling_experiments(data_constructor, args_list, num_iterations, profile_memory=False):
    results = []
    for kwargs in tqdm.tqdm(args_list):
        data = data_constructor(**kwargs)
        if profile_memory:
            memory = memory_usage((time_synthesis, (), {"data": data, "num_iterations": num_iterations}))
            max_memory = max(memory)
            results.append({**kwargs, "memory": max_memory})
        else:
            res_dict = time_synthesis(data=data, num_iterations=num_iterations)
            results.append({**kwargs, **res_dict})
    return pd.DataFrame(results)


@argh.arg("--dimensions", nargs="+", type=int)
@argh.arg("--categories", nargs="+", type=int)
@argh.arg("--jobs", nargs="+", type=str)
def main(num_iterations: int = 2500, default_dimensions: int = 100, default_size: int = 10000,
         default_categories: int = 10, out_dir: str = "", **kwargs):
    # test columns scaling
    arg_dict = {"dimensions": kwargs["dimensions"], "size": [default_size]}

    # -- continuous
    if "continuous_dim_time" in kwargs["jobs"]:
        args = generate_argument_combinations(arg_dict=arg_dict)
        continuous_times = run_profiling_experiments(data_constructor=create_multidimensional_gaussian,
                                                     args_list=args, num_iterations=num_iterations)
        dims = "_".join(map(str, arg_dict["dimensions"]))
        continuous_times.to_csv("{}/continuous-times-dim-{}.csv".format(out_dir, dims), index=False)

    if "continuous_dim_memory" in kwargs["jobs"]:
        args = generate_argument_combinations(arg_dict=arg_dict)
        continuous_times = run_profiling_experiments(data_constructor=create_multidimensional_gaussian,
                                                     args_list=args, num_iterations=num_iterations,
                                                     profile_memory=True)
        dims = "_".join(map(str, arg_dict["dimensions"]))
        continuous_times.to_csv("{}/continuous-memory-dim-{}.csv".format(out_dir, dims), index=False)

    # -- categorical
    if "categorical_dim_time" in kwargs["jobs"]:
        arg_dict["categories"] = [default_categories]
        args = generate_argument_combinations(arg_dict=arg_dict)
        categorical_times = run_profiling_experiments(data_constructor=create_multidimensional_categorical,
                                                      args_list=args, num_iterations=num_iterations)
        dims = "_".join(map(str, arg_dict["dimensions"]))
        categorical_times.to_csv("{}/categorical-times-dim-{}.csv".format(out_dir, dims), index=False)

    if "categorical_dim_memory" in kwargs["jobs"]:
        arg_dict["categories"] = [default_categories]
        args = generate_argument_combinations(arg_dict=arg_dict)
        categorical_times = run_profiling_experiments(data_constructor=create_multidimensional_categorical,
                                                      args_list=args, num_iterations=num_iterations,
                                                      profile_memory=True)
        dims = "_".join(map(str, arg_dict["dimensions"]))
        categorical_times.to_csv("{}/categorical-memory-dim-{}.csv".format(out_dir, dims), index=False)


    # test categories scaling
    if "categorical_scale_time" in kwargs["jobs"]:
        cat_arg_dict = {"dimensions": [default_dimensions], "size": [default_size],
                        "categories": kwargs["categories"]}
        args = generate_argument_combinations(arg_dict=cat_arg_dict)
        categories_scaling_times = run_profiling_experiments(data_constructor=create_multidimensional_categorical,
                                                             args_list=args, num_iterations=num_iterations)
        cats = "_".join(kwargs["categories"])
        categories_scaling_times.to_csv("{}/categories-scaling-times-cats-{}.csv".format(out_dir, cats), index=False)

    if "categorical_scale_memory" in kwargs["jobs"]:
        cat_arg_dict = {"dimensions": [default_dimensions], "size": [default_size],
                        "categories": kwargs["categories"]}
        args = generate_argument_combinations(arg_dict=cat_arg_dict)
        categories_scaling_times = run_profiling_experiments(data_constructor=create_multidimensional_categorical,
                                                             args_list=args, num_iterations=num_iterations,
                                                             profile_memory=True)
        cats = "_".join(kwargs["categories"])
        categories_scaling_times.to_csv("{}/categories-scaling-memory-cats-{}.csv".format(out_dir, cats), index=False)

    # profiling
    # -- continuous
    if "continuous_profile" in kwargs["jobs"]:
        filepath = "{}/profiler_continuous_dimensions_{}_size_{}.json".format(out_dir, default_dimensions, default_size)
        profiler_args = ProfilerArgs(filepath=filepath, period=num_iterations)
        data = create_multidimensional_gaussian(dimensions=default_dimensions, size=default_size)
        with BasicSynthesizer(data=data, profiler_args=profiler_args) as synthesizer:
            synthesizer.learn(data=data, num_iterations=num_iterations)

    # -- categorical
    if "categorical_profile" in kwargs["jobs"]:
        filepath = "{}/profiler_categorical_dimensions_{}_size_{}_categories_{}.json".format(out_dir,
                                                                                             default_dimensions,
                                                                                             default_size,
                                                                                             default_categories)
        profiler_args = ProfilerArgs(filepath=filepath, period=num_iterations)
        data = create_multidimensional_categorical(dimensions=default_dimensions, categories=default_categories,
                                                   size=default_size)
        with BasicSynthesizer(data=data, profiler_args=profiler_args) as synthesizer:
            synthesizer.learn(data=data, num_iterations=num_iterations)


if __name__ == "__main__":
    argh.dispatch_command(main)








