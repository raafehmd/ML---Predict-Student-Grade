import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tqdm.keras import TqdmCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import app_code
from app_code import read_data
import matplotlib.pyplot as plt
import os

# disabling bhosadpappu warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
pd.options.mode.chained_assignment = None  # default='warn'

# reading dataset
all_data = pd.read_csv('student-mat.csv', delimiter=';', skipinitialspace=True)
relev_data = all_data[
    ['age', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'higher',
     'internet',
     'romantic', 'freetime', 'goout', 'Dalc', 'health', 'absences', 'G1', 'G2', 'G3']]

# converting to 0s and 1s
relev_data['schoolsup'] = pd.Series(np.where(relev_data['schoolsup'].copy().values == 'yes', 1, 0),
                                    relev_data.index)
relev_data['famsup'] = pd.Series(np.where(relev_data['famsup'].copy().values == 'yes', 1, 0), relev_data.index)
relev_data['paid'] = pd.Series(np.where(relev_data['paid'].copy().values == 'yes', 1, 0), relev_data.index)
relev_data['activities'] = pd.Series(np.where(relev_data['activities'].copy().values == 'yes', 1, 0),
                                     relev_data.index)
relev_data['higher'] = pd.Series(np.where(relev_data['higher'].copy().values == 'yes', 1, 0), relev_data.index)
relev_data['internet'] = pd.Series(np.where(relev_data['internet'].copy().values == 'yes', 1, 0), relev_data.index)
relev_data['romantic'] = pd.Series(np.where(relev_data['romantic'].copy().values == 'yes', 1, 0), relev_data.index)

# classifying as data and labels
X = relev_data.drop(columns='G3')
y = relev_data['G3']

# splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# converting lists to nparray
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# loading model
model = models.load_model('model.h5')

# predicting using test data
stud_num = 78
trial_pred = X_test[[stud_num]]
prediction1 = model.predict(np.array(trial_pred))
print('\nStudent Data: \n {}'.format(trial_pred))
print('\n Predicted Final Grade: {}'.format(np.round(prediction1, 2)))
print('\n Actual Final Grade: {}'.format(y_test[stud_num]))

'''
# predicting user inputs
predict_this = [18, 3, 3, 2, 1, 0, 1, 0, 1, 1, 1, 3, 3, 0, 4, 36, 5, 13]
prediction = model.predict([predict_this])
print('\n\n' + str(prediction))
'''
