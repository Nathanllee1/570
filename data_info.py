import pandas as pd
import numpy as np

sample_data = pd.read_csv('data.csv')
X = sample_data[sample_data.columns[3:]]
y = sample_data[sample_data.columns[2]]

print(y.value_counts())