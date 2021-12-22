import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv('student-mat.csv', sep=';')
data = data[['G1', 'G2', 'G3', 'studytime', 'absences', 'failures']]

predict = 'G3'

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)


predictions= linear.predict(x_test)

print('\nData: [FirstGrade, SecondGrade, Studytime, absences, faliures]')
print('prediction: FinalGrade')

for x in range(len(predictions)):
    print('\n prediction: \n', predictions[x], '\n Data: \n', x_test[x], '\n actual result: \n', y_test[x], '\n -------------------------')