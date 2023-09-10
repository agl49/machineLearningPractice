# imports
from operator import index
import numpy as np
import pandas as pd
import seaborn as sea
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing

numbers = []

for i in range(0, 134):
    numbers.append([i])

indexArray = []

for i in range(0, 134):
    s = 'row' + str(i)
    indexArray.append(s)

print('numbers: ', numbers)
print('indexArray: ', indexArray)

X = pd.DataFrame(numbers,
    index=indexArray,
    columns=['colum']
                 )

thresh = 58.98

print('np.argwhere(X <= thresh): ', np.argwhere(X <= thresh))

print('np.argwhere(X <= argwhere).flatten()', 
      np.argwhere(X <= thresh).flatten())

print('X index: ', X.index)
print('X.colums', X.columns)



