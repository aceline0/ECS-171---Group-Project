import numpy as np
import seaborn as sns
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


df = pd.read_csv("../data/creditcard.csv")

df = df.drop_duplicates()
df = df.reset_index()
df = df.drop('index', axis=1)


df["Hour"] = [(s // 3600) % 24 for s in df["Time"]]


X = df.drop('Class', axis=1)
y = df['Class']

cols = list(df.columns)
cols.remove('Class')
scaler = MinMaxScaler()

df = pd.DataFrame(scaler.fit_transform(X), columns = cols)
df = pd.concat([df, y], axis=1)


df = df.drop(['Time', 'V13', 'V15', 'V22','V23','V24','V25','V26', 'V27','V28','Amount', 'V8', 'V21'], axis=1)


X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
smote = SMOTE(random_state=21)

X_train, y_train = smote.fit_resample(X, y)