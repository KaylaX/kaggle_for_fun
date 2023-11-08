# Import necessary libraries
import pandas as pd
from causalml.inference.meta import BaseXRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from causalml.match import NearestNeighborMatch

# Assume df is a pandas DataFrame with the following columns:
# 'treatment': binary indicator of whether the individual received the treatment (1) or not (0)
# 'outcome': the outcome variable of interest (e.g., earnings)
# 'confounders': other variables that may affect both treatment and outcome

# Split data into features and outcome
X = df.drop('outcome', axis=1)
y = df['outcome']

# Split the features into treatment and confounders
W = X['treatment']  # treatment assignment
X = X.drop('treatment', axis=1)  # confounders

# Propensity score matching
ps_model = RandomForestClassifier()
ps_model.fit(X, W)

# Estimate propensity scores
df['propensity_score'] = ps_model.predict_proba(X)[:, 1]

# Create a matching estimator using the NearestNeighborMatch
matcher = NearestNeighborMatch(replace=False)
# pairs each treated unit with one or more control units with the closest propensity scores
matched = matcher.match(data=df, treatment_col='treatment', score_cols=['propensity_score'])

# Calculate the Average Treatment Effect
ate = matched.groupby('treatment')['outcome'].mean().diff().iloc[-1]

print(f"The estimated ATE is {ate:.2f}")
