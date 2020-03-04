import argh
import numpy as np
import pandas as pd
import synthesized.testing.synthetic_distributions as dist


def main(output_dir: str = "data/time-series/synthetic",
         length: int = 1000, cycle_amplitude: float = 10.,
         cycle_period: float = 125, trend_slope: float = 2,
         trend_bias: float = 2, cont_ar_bias: float = 2,
         cont_ar_order: int = 10, cont_ar_acf: float = 2.,
         n_classes: int = 20,
         noise_sd: float = 1, num_series=10):
    # Make continuous data
    # -- Cyclical
    data = dist.create_time_series_data(func=dist.additive_sine(a=cycle_amplitude,
                                                                p=cycle_period,
                                                                sd=noise_sd),
                                        length=length)
    data.to_csv(f"{output_dir}/continuous-cyclical.csv", index=False)

    # -- Trend
    data = dist.create_time_series_data(func=dist.additive_linear(a=trend_slope,
                                                                  b=trend_bias,
                                                                  sd=noise_sd),
                                        length=length)
    data.to_csv(f"{output_dir}/continuous-trend.csv", index=False)

    # -- Auto-correlation
    phi = cont_ar_acf*np.ones(cont_ar_order)
    data = dist.create_time_series_data(func=dist.continuous_auto_regressive(phi=phi,
                                                                             c=cont_ar_bias,
                                                                             sd=noise_sd),
                                        length=length)
    data.to_csv(f"{output_dir}/continuous-autoregressive.csv", index=False)

    # -- Number of series
    phi = cont_ar_acf*np.ones(cont_ar_order)
    data_frames = []
    for i in range(num_series):
        local_data = dist.create_time_series_data(func=dist.continuous_auto_regressive(phi=phi,
                                                                                 c=cont_ar_bias,
                                                                                 sd=noise_sd),
                                                  length=length)
        local_data["id"] = i
        data_frames.append(local_data)
    data = pd.concat(data_frames)
    data.to_csv(f"{output_dir}/continuous-multiple-autoregressive.csv", index=False)

    # Make categorical data
    # -- Auto-correlation
    data = dist.create_time_series_data(func=dist.categorical_auto_regressive(n_classes=n_classes,
                                                                              sd=noise_sd),
                                        length=length)
    data.to_csv(f"{output_dir}/categorical-autoregressive.csv", index=False)

    # -- Number of series
    data_frames = []
    for i in range(num_series):
        local_data = dist.create_time_series_data(func=dist.categorical_auto_regressive(n_classes=n_classes,
                                                                                        sd=noise_sd),
                                                  length=length)
        local_data["id"] = i
        data_frames.append(local_data)
    data = pd.concat(data_frames)
    data.to_csv(f"{output_dir}/categorical-multiple-autoregressive.csv", index=False)


if __name__ == "__main__":
    argh.dispatch_command(main)

