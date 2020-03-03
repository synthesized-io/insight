import numpy as np
import pandas as pd

dataset  = pd.read_csv("transactions.csv")

dataset['date'] = pd.to_datetime(dataset['date'])

dataset1 = dataset[((dataset["date"] < pd.to_datetime("1993-02-01", format="%Y-%m-%d")) |
                    ((dataset["date"] >= pd.to_datetime("1993-03-01", format="%Y-%m-%d")) & (dataset["date"] < pd.to_datetime("1993-04-01", format="%Y-%m-%d"))) |
                    ((dataset["date"] >= pd.to_datetime("1993-05-01", format="%Y-%m-%d")) & (dataset["date"] < pd.to_datetime("1993-06-01", format="%Y-%m-%d"))) |
                    ((dataset["date"] >= pd.to_datetime("1993-07-01", format="%Y-%m-%d")) & (dataset["date"] < pd.to_datetime("1993-08-01", format="%Y-%m-%d"))) |
                    ((dataset["date"] >= pd.to_datetime("1993-09-01", format="%Y-%m-%d")) & (dataset["date"] < pd.to_datetime("1993-10-01", format="%Y-%m-%d"))) |
                    ((dataset["date"] >= pd.to_datetime("1993-11-01", format="%Y-%m-%d")) & (dataset["date"] < pd.to_datetime("1993-12-01", format="%Y-%m-%d"))))]

dataset2 = dataset[((dataset["date"] >= pd.to_datetime("1993-02-01", format="%Y-%m-%d")) & (dataset["date"] < pd.to_datetime("1993-03-01", format="%Y-%m-%d"))) |
                    ((dataset["date"] >= pd.to_datetime("1993-04-01", format="%Y-%m-%d")) & (dataset["date"] < pd.to_datetime("1993-05-01", format="%Y-%m-%d"))) |
                    ((dataset["date"] >= pd.to_datetime("1993-06-01", format="%Y-%m-%d")) & (dataset["date"] < pd.to_datetime("1993-07-01", format="%Y-%m-%d"))) |
                    ((dataset["date"] >= pd.to_datetime("1993-08-01", format="%Y-%m-%d")) & (dataset["date"] < pd.to_datetime("1993-09-01", format="%Y-%m-%d"))) |
                    ((dataset["date"] >= pd.to_datetime("1993-10-01", format="%Y-%m-%d")) & (dataset["date"] < pd.to_datetime("1993-11-01", format="%Y-%m-%d"))) |
                    ((dataset["date"] >= pd.to_datetime("1993-12-01", format="%Y-%m-%d")) & (dataset["date"] < pd.to_datetime("1994-01-01", format="%Y-%m-%d")))]

d1 = {1:1, 3:2, 5:3, 7:4, 9:5, 11:6}
d2 = {2:1, 4:2, 6:3, 8:4, 10:5, 12:6}

import datetime

def check_date(year, month, day):
    correctDate = None
    try:
        newDate = datetime.datetime(year, month, day)
        correctDate = True
    except ValueError:
        correctDate = False
    return correctDate

dataset2["date"] = dataset2["date"].apply(lambda dt: dt.replace(month = d2[dt.month])
                                          if check_date(dt.year, d2[dt.month], dt.day) == True else None)

dataset1["date"] = dataset1["date"].apply(lambda dt: dt.replace(month = d1[dt.month])
                                          if check_date(dt.year, d1[dt.month], dt.day) == True else None)

d = {}
for i in range(len(dataset2.account_id.unique())):
    d[dataset2.account_id.unique()[i]] = i

dataset2["account_id"] = dataset2["account_id"].apply(lambda x: d[x])

d = {}
for i in range(len(dataset2.trans_id.unique())):
    d[dataset2.trans_id.unique()[i]] = i

dataset2["trans_id"] = dataset2["trans_id"].apply(lambda x: d[x])

import ast

dataset2["mean_income"] = dataset2["mean_income"].apply(lambda x: str([round(float(ast.literal_eval(x)[0])
                                                                       + np.random.normal(10, 10, 1)[0],2) ]))


dataset2["amount"] = dataset2["amount"].apply(lambda x: str(float(x) + np.random.choice(range(-2,2,1), 1)[0]))
dataset2["balance"] = dataset2["balance"].apply(lambda x: str( round(float(x),2) + np.random.choice(range(-20,20,1), 1)[0]))


dataset2 = dataset2[dataset2.date.notnull()]
dataset1 = dataset1[dataset1.date.notnull()]

dataset2 = dataset2.reset_index(drop = True)
dataset1 = dataset1.reset_index(drop = True)

dataset1.to_csv("demo_original_dataset.tsv", sep='\t')
dataset2.to_csv("demo_synthetic_dataset_perfect.tsv", sep='\t')

print(dataset1.head(10))
print(dataset2.head(10))


