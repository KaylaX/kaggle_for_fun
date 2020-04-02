import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb


train=pd.read_csv('train_new.csv')
newdata = train

feature_all = newdata[newdata.columns.difference(['Unnamed: 0','Id', 'groupId','matchId','matchType','winPoints',
                                                'winPlacePerc'])]
target_all = newdata['winPlacePerc']

### split the training dataset to two parts: train and validate data. the ratio is 8:2
train_all_index =  round(int(len(newdata.matchId.unique())*0.8))

train_newdata = newdata.loc[newdata['matchId'].isin(newdata.matchId.unique()[:train_all_index])]
validate_newdata =  newdata.loc[newdata['matchId'].isin(newdata.matchId.unique()[train_all_index:])]


# remove useless variable
x_train = train_newdata[train_newdata.columns.difference(['Unnamed: 0','Id', 'groupId','matchId', 'matchType','winPoints',
                                                'winPlacePerc'])]
y_train = train_newdata['winPlacePerc']

x_val = validate_newdata[validate_newdata.columns.difference(['Unnamed: 0','Id', 'groupId','matchId', 'matchType','winPoints',
                                                'winPlacePerc'])]
y_val = validate_newdata['winPlacePerc']

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'mse',
    'metric': 'mse',
    'num_leaves': 40,
    'max_depth':8,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8
}
lgb_train= lgb.Dataset(x_train, y_train)
lgb_val = lgb.Dataset(x_val, y_val)
lgb_model = lgb.train(lgb_params, lgb_train, valid_sets=[lgb_train, lgb_val])


#pred_test_y = model.predict(x_test, num_iteration=model.best_iteration)

## view the feature importance
plt.figure(figsize=(16,12))
lgb.plot_importance(lgb_model, max_num_features=40)
plt.title("Featurertances")
plt.show()

