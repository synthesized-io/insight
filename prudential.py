import warnings
warnings.filterwarnings(action='ignore', message='numpy.dtype size changed')

from datetime import datetime
import pandas as pd
from synthesized.core import BasicSynthesizer

data = pd.read_csv('data/prudential_train.csv')

# Id: A unique identifier associated with an application.
data = data.drop(labels='Id', axis=1)

# Medical_Keyword_1-48 are dummy variables.
data = data.drop(labels=['Medical_Keyword_' + str(n) for n in range(1, 49)], axis=1)


# The following variables are all categorical (nominal):
for attribute in ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9', 'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40', 'Medical_History_41']:
    data[attribute] = data[attribute].astype(dtype='category')
    # print(attribute, data[attribute].dtype.categories)

# The following variables are continuous:
for attribute in ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1']:
    data[attribute] = data[attribute].astype(dtype='float32')

for attribute in ['Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']:
    data = data.drop(labels=attribute, axis=1)

# The following variables are discrete:
for attribute in ['Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32']:
    data = data.drop(labels=attribute, axis=1)
    # print(attribute, data[attribute].isna().sum())
    # data[attribute] = data[attribute].astype(dtype='category')
    # # print(attribute, data[attribute].dtype.categories)

print(len(data))
data = data.dropna()
print(len(data))
print(data.head(5))

# Response: This is the target variable, an ordinal variable relating to the final decision associated with an application
data['Response'] = data['Response'].astype(dtype='category')
# print('Response', data['Response'].dtype.categories)


with BasicSynthesizer(dtypes=data.dtypes) as synthesizer:
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    synthesizer.learn(data=data, verbose=5000)
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
    synthesized = synthesizer.synthesize(n=10000)
    print(synthesized.head(10))
    print(datetime.now().strftime('%H:%M:%S'), flush=True)
