import numpy as np
import pandas as pd

train = pd.read_csv('./train_V2.csv')
test = pd.read_csv('./test_V2.csv')

train.isnull().any()

pd.set_option('display.max_columns', None)
train.head(10)

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

### feature engineering

train = train.dropna()
train['playersJoined'] = train.groupby('matchId')['Id'].transform('count')

train = train.reset_index(drop=True)

# boost+heal
train['total_heal'] = train['boosts'] + train['heals']

# matchType, divide into seven categories solo, duo, squad, solo-fpp, duo-fpp, squad-fpp and others
train['squad-fpp'] = [1 if train['matchType'][i] == 'squad-fpp' else 0 for i in range(0, len(train))]
train['duo'] = [1 if train['matchType'][i] == 'duo' else 0 for i in range(0, len(train))]

train['solo-fpp'] = [1 if train['matchType'][i] == 'solo-fpp' else 0 for i in range(0, len(train))]
train['squad'] = [1 if train['matchType'][i] == 'squad' else 0 for i in range(0, len(train))]

train['duo-fpp'] = [1 if train['matchType'][i] == 'duo-fpp' else 0 for i in range(0, len(train))]
train['solo'] = [1 if train['matchType'][i] == 'solo' else 0 for i in range(0, len(train))]

train['others'] = [1 if train['matchType'][i] not in ('squad-fpp', 'duo', 'solo-fpp', 'squad', 'duo-fpp', 'solo'
                                                      ) else 0 for i in range(0, len(train))]

# total distance. In there, I consider two situations: equal-weight and not. I assigned less weight in rideDistance
# since I thought drive a car has a higher speed. such distance can't reflect whether this player is a defender or guard
train['totalDistance_equalweight'] = train['walkDistance'] + train['swimDistance'] + train['rideDistance']
train['totalDistance_1:1:0.5'] = train['walkDistance'] + train['swimDistance'] + 0.5 * train['rideDistance']

# total attack. same as total distance. assign less weight for assists and DBNOs
train['totalattack_equalweight'] = train['kills'] + train['DBNOs'] + train['assists']
train['totalattack_1:.8:.5'] = train['kills'] + 0.8 * train['DBNOs'] + 0.5 * train['assists']

# walkDistance
# train['walkdisCategory']=pd.cut(train['walkDistance'],
#                                [-1,686,26000],
#                               labels=['guard','defender'])

# kill over distance
train['killsOverWalkDistance'] = [train['kills'][i] if train['walkDistance'][i] == 0
                                  else train['kills'][i] / train['walkDistance'][i] for i in range(0, len(train))]
train['killsOverDistance_equalweight'] = [train['kills'][i] if train['totalDistance_equalweight'][i] == 0
                                          else train['kills'][i] / train['totalDistance_equalweight'][i] for i in
                                          range(0, len(train))]
train['killsOverDistance_1:1:0.5'] = [train['kills'][i] if train['totalDistance_1:1:0.5'][i] == 0
                                      else train['kills'][i] / train['totalDistance_1:1:0.5'][i] for i in
                                      range(0, len(train))]

# headshot rate

train['headshotrate'] = [0 if train['kills'][i] == 0 else train['headshotKills'][i] / train['kills'][i] for i in
                         range(0, len(train))]

# normalization. normalize the variable based on the # of plyers in each match.
# but seems like not reasonable enough. take kills for example, if there are more player, it is true that the
# maximum # you could kill increase, however, the risk you will be killed also increase
train['killNorm'] = train['kills'] / train['playersJoined']
train['damageDealtNorm'] = train['damageDealt'] / train['playersJoined']
train['assistNorm'] = train['assists'] / train['playersJoined']
train['DBNONorm'] = train['DBNOs'] / train['playersJoined']
train['totalattack_equalweightNorm'] = train['totalattack_equalweight'] / train['playersJoined']
train['totalattack_1:.8:.5Norm'] = train['totalattack_1:.8:.5'] / train['playersJoined']
train['weaponsNorm'] = train['weaponsAcquired'] / train['playersJoined']
train['matchDurationNorm'] = train['matchDuration'] / train['playersJoined']

# binning
# train['killsCount']=pd.cut(train['killNorm'],[-1,0,1,2,3,4,130],
#                           labels=['0_kills','1_kills','2_kills',
#                                  '3_kills','4_kills','4+_kills'])

# train['rank']=pd.cut(train['rankPoints'],[-2,1,1500,6000],
#                           labels=['rank3','rank2','rank1'])


# group var. view players in a team. whether the team is afffected by the best or worst player

train['totalKills_team'] = train.groupby(['matchId', 'groupId'])['kills'].transform('sum')
train['minKills_team'] = train.groupby(['matchId', 'groupId'])['kills'].transform('min')
train['maxKills_team'] = train.groupby(['matchId', 'groupId'])['kills'].transform('max')
train['meanKills_team'] = train.groupby(['matchId', 'groupId'])['kills'].transform('mean')

train['totalDamage_team'] = train.groupby(['matchId', 'groupId'])['damageDealt'].transform('sum')
train['minDamage_team'] = train.groupby(['matchId', 'groupId'])['damageDealt'].transform('min')
train['maxDamage_team'] = train.groupby(['matchId', 'groupId'])['damageDealt'].transform('max')
train['meanDamage_team'] = train.groupby(['matchId', 'groupId'])['damageDealt'].transform('mean')

train['totalHeals_team'] = train.groupby(['matchId', 'groupId'])['total_heal'].transform('sum')
train['minHeals_team'] = train.groupby(['matchId', 'groupId'])['total_heal'].transform('min')
train['maxHeals_team'] = train.groupby(['matchId', 'groupId'])['total_heal'].transform('max')
train['meanHeals_team'] = train.groupby(['matchId', 'groupId'])['total_heal'].transform('mean')

train['totalDistiance_equalweight_team'] = train.groupby(['matchId', 'groupId'])['totalDistance_equalweight'].transform(
    'sum')
train['minDistiance_equalweight_team'] = train.groupby(['matchId', 'groupId'])['totalDistance_equalweight'].transform(
    'min')
train['maxDistiance_equalweight_team'] = train.groupby(['matchId', 'groupId'])['totalDistance_equalweight'].transform(
    'max')
train['meanDistiance_equalweight_team'] = train.groupby(['matchId', 'groupId'])['totalDistance_equalweight'].transform(
    'mean')

train['totalDistance_1:1:0.5_team'] = train.groupby(['matchId', 'groupId'])['totalDistance_1:1:0.5'].transform('sum')
train['minDistance_1:1:0.5_team'] = train.groupby(['matchId', 'groupId'])['totalDistance_1:1:0.5'].transform('min')
train['maxDistance_1:1:0.5_team'] = train.groupby(['matchId', 'groupId'])['totalDistance_1:1:0.5'].transform('max')
train['meanDistance_1:1:0.5_team'] = train.groupby(['matchId', 'groupId'])['totalDistance_1:1:0.5'].transform('mean')

train['totalWeaponsAcquired_team'] = train.groupby(['matchId', 'groupId'])['weaponsAcquired'].transform('sum')
train['minWeaponsAcquired_team'] = train.groupby(['matchId', 'groupId'])['weaponsAcquired'].transform('min')
train['maxWeaponsAcquired_team'] = train.groupby(['matchId', 'groupId'])['weaponsAcquired'].transform('max')
train['meanWeaponsAcquired_team'] = train.groupby(['matchId', 'groupId'])['weaponsAcquired'].transform('mean')

train['minKillPlace_team'] = train.groupby(['matchId', 'groupId'])['killPlace'].transform('min')
train['maxKillPlace_team'] = train.groupby(['matchId', 'groupId'])['killPlace'].transform('max')
train['meanKillPlace_team'] = train.groupby(['matchId', 'groupId'])['killPlace'].transform('mean')

train.to_csv('train_new.csv')