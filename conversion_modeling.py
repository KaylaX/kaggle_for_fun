import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # splitting the data
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as dtc # tree algorithm
from sklearn.tree import plot_tree # tree diagram

df=pd.read_csv('data.csv')

df.head()

df.count()/len(df)

len(df[df['converted']==1]) #64076

len(df) # 919084

df.columns

X=df[['treatment_name', 'gv_band_a', 'gv_band_b', 'gv_band_c', 'gv_band_d',
       'weekday', 'was_price_flag', 'discount_band_0_10',
       'discount_band_10_19', 'discount_band_19_29', 'price_band_0_26',
       'price_band_26_68', 'price_band_68_149', 'price_band_149_343']]
y=df['converted']

############ split into training and testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

############ Logistic Regression

import imblearn
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy=0.5)

# fit and apply the transform
X_over, y_over = undersample.fit_resample(X_train, y_train)

# summarize class distribution
print(Counter(y_over))

lr = LogisticRegression()
lr.fit(X_over, y_over)

# Predicting on the test data
y_pred_log = lr.predict(X_test)

#Calculating and printing the f1 score
f1_test_log = metrics.f1_score(y_test, y_pred_log)
print('The f1 score for the testing data:', f1_test_log)

# Print the confusion matrix
y_pred_train = lr.predict(X_over)
print(metrics.classification_report(y_over, y_pred_train, digits=3))
print(metrics.confusion_matrix(y_over, y_pred_train))

print(metrics.confusion_matrix(y_test, y_pred_log))

print(metrics.classification_report(y_test, y_pred_log, digits=3))

############# Decision Tree

# Create Decision Tree classifier object
clf = dtc(criterion="entropy", max_depth=5, min_weight_fraction_leaf=0.1, ccp_alpha=0.001)

# Train Decision Tree Classifier
clf = clf.fit(X_over,y_over)

#Predict the response for test dataset
y_pred_dt = clf.predict(X_test)

print('The f1 score for the testing data:', metrics.f1_score(y_test, y_pred_dt))

# Print the confusion matrix
y_pred_train_clf = clf.predict(X_over)
print(metrics.classification_report(y_over, y_pred_train_clf, digits=3))
print(metrics.confusion_matrix(y_over, y_pred_train_clf))

print(metrics.confusion_matrix(y_test, y_pred_dt))

print(metrics.classification_report(y_test, y_pred_dt, digits=3))

############## Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=4, random_state=0)
rf.fit(X_over, y_over)

#Predict the response for test dataset
y_pred_rf = rf.predict(X_test)

print('The f1 score for the testing data:', metrics.f1_score(y_test, y_pred_rf))

# Print the confusion matrix
y_pred_train_rf = rf.predict(X_over)
print(metrics.classification_report(y_over, y_pred_train_rf, digits=3))
print(metrics.confusion_matrix(y_over, y_pred_train_rf))

print(metrics.confusion_matrix(y_test, y_pred_rf))

print(metrics.classification_report(y_test, y_pred_rf, digits=5))