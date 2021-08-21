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


X=df[['treatment_name','gv_band_a', 'gv_band_b', 'gv_band_c', 'gv_band_d',
       'was_price_flag',
       'discount_band_0_10','discount_band_10_19', 'discount_band_19_29',
       'price_band_0_26',
       'price_band_26_68', 'price_band_68_149', 'price_band_149_343'
        ]]
y=df['converted']

############ Logistic Regression, Decision Tree, Random Forest

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# initializing classifiers
lr = LogisticRegression(random_state=42, max_iter=150)
clf = dtc(criterion="entropy", max_depth=5, min_weight_fraction_leaf=0.1)
rf = RandomForestClassifier(max_depth=4, random_state=0)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
score_lr =[]
score_dt=[]
score_rf=[]
pred_test_lr =0
pred_test_dt=0
pred_test_rf=0
i=1

for train_index,test_index in skf.split(X,y):
       print('{} of KFold {}'.format(i,skf.n_splits))
       x_train,x_test = X.loc[train_index],X.loc[test_index]
       y_train,y_test = y.loc[train_index],y.loc[test_index]

       lr.fit(x_train, y_train)
       clf.fit(x_train, y_train)
       rf.fit(x_train, y_train)

       y_pred_log = lr.predict(x_test)
       y_pred_dt = clf.predict(x_test)
       y_pred_rf = rf.predict(x_test)

       f1_test_lr = metrics.f1_score(y_test, y_pred_log)
       f1_test_dt = metrics.f1_score(y_test, y_pred_dt)
       f1_test_rf = metrics.f1_score(y_test, y_pred_rf)
       print("Logistic regression F1 score:{}".format(f1_test_lr))
       print("Decision tree F1 score:{}".format(f1_test_dt))
       print("Random Forest F1 score:{}".format(f1_test_rf))

       score_lr.append(f1_test_lr)
       score_dt.append(f1_test_dt)
       score_rf.append(f1_test_rf)

       i+=1