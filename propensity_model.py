from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

# original ratio

# defining the classifier
xgboost = XGBClassifier(booster='gbtree', 
                        objective= 'binary:logistic',
                        eval_metric = 'aucpr')

# Splitting data
df_has_seat_band = df_all[df_all['current_seat_active']!=0]
train_data, val_data = train_test_split(df_has_seat_band, test_size=0.4, shuffle=True)
train_X = train_data[features]
train_y = train_data['substance_flag']
val_X = val_data[features]
val_y = val_data['substance_flag']

# define pipeline
steps = [('model', xgboost)]
pipeline_xgboost_1_200 = Pipeline(steps=steps)

# Setup grid of parameters
param_dist = {
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'model__n_estimators': [50, 100, 150, 200],
    'model__max_depth': [3, 4, 5, 6, 7],
    'model__gamma': [0, 1, 2, 3, 4],
    'model__subsample': [0.6, 0.7, 0.8, 0.9],
    'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'model__min_child_weight': [1, 2, 3, 4, 5],
    'model__scale_pos_weight': [10, 20, 30, 40, 50]
}

# initialize RandomizedSearchCV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)

from sklearn.metrics import make_scorer, average_precision_score

# Create a custom scorer
custom_scorer = make_scorer(average_precision_score, needs_proba=True)

# Initialize RandomizedSearchCV with custom scorer
pipeline_xgboost_1_200 = RandomizedSearchCV(pipeline_xgboost_1_200, param_distributions=param_dist, n_iter=20, scoring=custom_scorer, cv=cv, n_jobs=2, verbose=3)

# Fit RandomizedSearchCV
pipeline_xgboost_1_200.fit(train_X, train_y)

# Display best parameters and corresponding score
print("Best parameters found: ", pipeline_xgboost_1_200.best_params_)
print(f"Best average precision score (AUC-PR approximation): {pipeline_xgboost_1_200.best_score_:.3f}")

print("Best cross-validation F1 Score: {:.3f}".format(pipeline_xgboost_1_200.best_score_))

# Save the best model as the original xgboost model for later use with SHAP
original_xgboost_1_200 = pipeline_xgboost_1_200.best_estimator_.named_steps['model']

# predict probabilities on validation data
val_y_pred_prob = pipeline_xgboost_1_1.best_estimator_.predict_proba(val_X)[:, 1]

# calculate the quantile cutoff point
cutoff_quantile = np.quantile(val_y_pred_prob, q=1 - np.mean(y_train_under))

print("Quantile Cutoff Point: ", cutoff_quantile)

# classify predictions based on the quantile cutoff point
val_y_pred = np.where(val_y_pred_prob > cutoff_quantile, 1, 0)