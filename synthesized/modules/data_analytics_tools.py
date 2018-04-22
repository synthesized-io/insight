from __future__ import division, print_function, absolute_import

import random
import warnings

import numpy as np
import pandas as pd


def suppress_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return wrapper


@suppress_warnings
def show_data(array_a, array_b=None, array_c=None, nrow=2, ncol=10, figsize=None, save_loc=None):
    # import without warnings
    from matplotlib import pyplot as plt

    # if both are None, just plot one
    if array_b is None and array_c is None:
        nrow = 1

    # if kw specifically makes B None, shift it over
    elif array_b is None:
        array_b = array_c
        array_c = None
        nrow = 2

    # otherwise if just plotting the first two...
    elif array_c is None:
        nrow = 2

    elif array_b is not None and array_c is not None:
        nrow = 3

    if nrow not in (1, 2, 3):
        raise ValueError('nrow must be in (1, 2)')

    if figsize is None:
        figsize = (ncol, nrow)

    f, a = plt.subplots(nrow, ncol, figsize=figsize)
    arrays = [array_a, array_b, array_c]

    def _do_show(the_figure, the_array):
        the_figure.imshow(the_array)
        the_figure.axis('off')

    for i in range(ncol):
        if nrow > 1:
            for j in range(nrow):
                _do_show(a[j][i], np.reshape(arrays[j][i], (16, 8)))
        else:
            _do_show(a[i], np.reshape(array_a[i], (16, 8)))

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    f.show()
    plt.draw()

    # if save...
    if save_loc is not None:
        plt.savefig(save_loc)


class DataPipeline():
    def __init__(self):
        self.working_dataset = None
        #specified by users
        self.original_dataset = None
        #original dataset is given to us
        self.training_dataset = []
        #internal hidden usage, hidden from users

    # def load_from_csv(self, path):
    # try to read from csv
    #	self.targetDataSet = pd.DataFrame.from_csv(path,sep='\t')

    def load_from_csv(self, path):
        # try to read from csv
        self.original_dataset  = pd.read_csv(path)

    def preprocess_crude_dataset(self, feature_list, dataset=None):
        # try to read from csv
        if dataset != None:
            self.original_dataset = dataset
        import numpy as np
        df = self.original_dataset[feature_list]
        for feature in feature_list:
            try:
                df = df[np.isfinite(df[feature])]
            except:
                continue
        # len(distinct_map.keys())
        df['date'] = pd.to_datetime(df['date'])
        self.working_dataset = df[((df["date"] < pd.to_datetime("1994-01-01", format="%Y-%m-%d")))]

    def load_from_sql(self, path):
        pass

    # read from sql

    def prepare_dataset_for_training(self):
        self.working_dataset['value'] = self.working_dataset['value'] / 10000
        self.working_dataset['date'] = pd.to_datetime(self.working_dataset['date'])

        spendings_clusters = 4
        transactions_time_resolution = 32
        transactions_time_resolution_with_padding = 32
        unique_ids = self.working_dataset.seqId.unique()
        months = np.arange(1, 13)

        for ids in unique_ids:
            for month in months:
                v = pd.DataFrame({}, index=range(0, spendings_clusters),
                                 columns=range(1, transactions_time_resolution + 1))
                # e2 = pd.DataFrame({}, index = range(0,spendings_clusters), columns = range(-2,1))
                # e3 = pd.DataFrame({}, index = range(0,spendings_clusters), columns = range(8,10))
                # e2 = e2.fillna(0)
                # e3 = e3.fillna(0)
                v = v.fillna(0.)
                fw = self.working_dataset[
                    (self.working_dataset["seqId"] == ids) & (self.working_dataset["date"].dt.month == month)]
                fw.reset_index(inplace=True)
                if (fw.shape[0] != 0):
                    for i in range(fw.shape[0]):
                        v.set_value(int(fw["category"][i]), int(fw["date"][i].day), fw["value"][i])
                    self.training_dataset.append(v.values.flatten())

    def get_outlier_dataset(self):
        # should be made more robust
        first_cluster_of_customers = self.working_dataset["seqId"].unique()[1:101]
        second_cluster_of_customers = self.working_dataset["seqId"].unique()[101:1001]

        spending_first_cluster = np.random.normal(100000, 10000, 100)
        spending_second_cluster = np.random.normal(100000, 10000, 900)

        spending_first_cluster = np.concatenate([[i] * 12 for i in spending_first_cluster], axis=0)
        spending_second_cluster = np.concatenate([[i] * 12 for i in spending_second_cluster], axis=0)

        outlier_dataset = pd.DataFrame({}, columns=self.working_dataset.columns)
        outlier_dataset['seqId'] = np.concatenate(
            [np.repeat(i, 12) for i in [first_cluster_of_customers, second_cluster_of_customers]], axis=0)
        outlier_dataset['category'] = np.repeat(1, 12 * 1000)
        outlier_dataset['value'] = np.concatenate([spending_first_cluster, spending_second_cluster], axis=0)
        start = pd.to_datetime("1993-01-01", format="%Y-%m-%d")
        end = pd.to_datetime("1993-12-30", format="%Y-%m-%d")
        outlier_dataset['date'] = np.concatenate([pd.date_range(start, end, freq='MS')] * 1000, axis=0)

        outlierDataSet = self.working_dataset[~((self.working_dataset["date"].
                                                 isin(pd.date_range(start, end, freq='MS')))
                                                & (self.working_dataset["category"] == 1)
                                                )]
        outlierDataSet = pd.concat([outlierDataSet, outlier_dataset])
        return (outlierDataSet)

    def get_aggregated_dataset_for_snapshot_generation(self, targetDataSet, multiplier, spendings_clusters):

        transactions_time_resolution = 32
        unique_ids = targetDataSet.seqId.unique()
        training_set = []
        months = np.arange(1, 13)

        for ids in unique_ids:
            v = pd.DataFrame({}, columns=range(0, spendings_clusters * multiplier),
                             index=range(1, transactions_time_resolution + 1))
            v = v.fillna(0.)
            for month in months:
                fw = targetDataSet[(targetDataSet["seqId"] == ids) & (targetDataSet["date"].dt.month == month)]
                fw.reset_index(inplace=True)
                if (fw.shape[0] != 0):
                    for i in range(fw.shape[0]):
                        v.set_value(int(fw["date"][i].day), int(fw["category"][i]) * multiplier + (month % multiplier),
                                    fw["value"][i])
                if (month % multiplier == 0):
                    training_set.append(v.values.flatten())
                    v = pd.DataFrame({}, columns=range(0, spendings_clusters * multiplier),
                                     index=range(1, transactions_time_resolution + 1))
                    v = v.fillna(0)
        return (np.asarray(training_set))

    def compress_dataset(seld, dataSet, multiplier, clusters_number):
        newDataSet = []
        for dataSetInstance in dataSet:
            a = pd.DataFrame(data=dataSetInstance, index=np.arange(1, 33),
                             columns=np.arange(1, clusters_number * multiplier + 1))
            b = pd.DataFrame({}, index=np.arange(1, 33), columns=np.arange(1, clusters_number + 1))
            for i in np.arange(1, a.shape[0] + 1):
                for k in np.arange(1, b.shape[1] + 1):
                    b[k][i] = random.choice([a[(k - 1) * multiplier + 1][i], a[(k - 1) * multiplier + 2][i],
                                             a[(k - 1) * multiplier + 3][i]])
            newDataSet.append(b)
        return (newDataSet)

    def get_dataset_batch(self, months):
        spendings_clusters = 4
        transactions_time_resolution = 32
        transactions_time_resolution_with_padding = 32
        unique_ids = self.working_dataset.seqId.unique()
        dataset_batch = []

        for ids in unique_ids:
            for month in months:
                v = pd.DataFrame({}, index=range(0, spendings_clusters),
                                 columns=range(1, transactions_time_resolution + 1))

            # e2 = pd.DataFrame({}, index = range(0,spendings_clusters), columns = range(-2,1))
            # e3 = pd.DataFrame({}, index = range(0,spendings_clusters), columns = range(8,10))
            # e2 = e2.fillna(0)
            # e3 = e3.fillna(0)
            v = v.fillna(0.)
            fw = self.working_dataset[
                (self.working_dataset["seqId"] == ids) & (self.working_dataset["date"].dt.month == month)]
            fw.reset_index(inplace=True)

            if (fw.shape[0] != 0):
                for i in range(fw.shape[0]):
                    v.set_value(int(fw["category"][i]), int(fw["date"][i].day), fw["value"][i])
                #  v = pd.concat([e2, v, e3], axis=1)
                dataset_batch.append(v.values.flatten())
        return (np.asarray(dataset_batch))

    def show_data(self):
        show_data(self.training_dataset)

    # do we need this?
    def clean(self):
        pass
