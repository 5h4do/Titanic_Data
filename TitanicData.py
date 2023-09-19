from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf

import tensorflow.feature_column as fc

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

print(dftrain.head()) # To show it in a cooler structure

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# print(dftrain.head()) # after printing this one because is after both y_train and y_eval are printed the survival data is deleted from the training data output

# print(dftrain.loc[0], y_train.loc[0]) # by using this we can call the individual and check its info (Name : 0 means didnt survive)

# print(dftrain.describe())  # give us the overall INFO like average of something

# print(dftrain.shape) # 627 rows and 9 columns


# dftrain.age.hist(bins=20)
# dftrain.sex.value_counts().plot(kind='barh') # how many male and female were travelling

# dftrain['class'].value_counts().plot(kind='barh') # which class they were assigned and how many

pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive') # the mean of how many male and female survived in percentage

# Show the plot
plt.show() 
