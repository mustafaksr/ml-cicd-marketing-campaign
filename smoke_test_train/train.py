import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectFromModel, f_classif, SelectKBest, VarianceThreshold
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier ,HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, cross_val_predict ,train_test_split
import warnings
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))


project_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_dir)

from common.utils import *
warnings.filterwarnings("ignore")
df = pd.read_csv(os.path.join(os.getcwd(),"train/bank.csv")).sample(frac=0.25,random_state=42)
X, X_test, y, y_test = split_data(df)
X = add_features(X)
X_test = add_features(X_test)

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Define the preprocessing steps for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
#    ("o", OrdinalEncoder())
])

# Create a ColumnTransformer that applies the appropriate transformations to each feature type
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


xgb_params = {'n_estimators': 150,
  'min_child_weight': 3,
  'max_depth': 3,
  'learning_rate': 0.1}
lgb_params = {'colsample_bytree': 0.7774799983649324, 'learning_rate': 0.007653648135411494, 'max_depth': 5,
              'n_estimators': 350, 'reg_alpha': 0.14326300616140863, 'reg_lambda': 0.9310129332502252,
              'subsample': 0.6189257947519665,}
cat_params = {'random_strength': 0.1, 'one_hot_max_size': 10, 'max_bin': 100, 'learning_rate': 0.01,
              'l2_leaf_reg': 0.5, 'grow_policy': 'Lossguide', 'depth': 5, 'bootstrap_type': 'Bernoulli','verbose':False,
           }
models=[
    ("bag",BaggingClassifier(SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber",n_jobs=-1),n_jobs=-1)),
    ("bag2",BaggingClassifier(LogisticRegression(solver='sag',n_jobs=-1))),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(**xgb_params)),
    ('lgb', LGBMClassifier(**lgb_params)),
    ("log", LogisticRegression(solver='sag',n_jobs=-1)),
    ("sgd", SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber",n_jobs=-1)),
   ('rf', RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1)),
    ('cat', CatBoostClassifier(**cat_params)),
]

X_t = preprocessor.fit_transform(X)
scr_list = pd.DataFrame()
oof_list = pd.DataFrame()
for (lbl, mod) in models:
    score =  np.mean(cross_val_score(Pipeline(steps=[
        #https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
        ('feature_selection', VarianceThreshold(threshold=(.85 * (1 - .85)))),
        ('model', mod)]),X_t,y,n_jobs=-1,cv=5))
    scr_list.loc[lbl,"score"] = score
    val_preds = cross_val_predict(Pipeline(steps=[
        #https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
        ('feature_selection', VarianceThreshold(threshold=(.85 * (1 - .85)))),
        ('model', mod)]),X_t,y,method="predict_proba",n_jobs=-1,cv=5)[:,1]
    oof_list[lbl] = val_preds
    print(lbl,score)

model_weights0 = RidgeClassifier(random_state = 1).fit(oof_list, y).coef_[0]
df_model_weights = pd.DataFrame(model_weights0, index=list(oof_list), columns=["Weight / Model"])
df_model_weights_sorted = df_model_weights.sort_values(by="Weight / Model", ascending=False)
print(df_model_weights_sorted)

# Create the full pipeline for the voting classifier
voting_model = VotingClassifier(estimators=models,weights=model_weights0,
    voting='soft',n_jobs = -1)

# Use Stratified K-Fold cross-validation
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Use Brier Score Loss as the evaluation metric
brier_scores = []
test_predictions_accumulated = np.zeros(len(X_test))
np.random.seed(1)
for fold, (train_idx, valid_idx) in enumerate(stratified_kfold.split(X, y), start=1):
    # Split the data
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # Apply preprocessing and feature engineering
    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    test_data_transformed = preprocessor.transform(X_test)

    # Create the final pipeline
    pipeline = Pipeline(steps=[
        #https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
        ('feature_selection', VarianceThreshold(threshold=(.85 * (1 - .85)))),
        
        ('model', voting_model)])

    # Fit the model to the training data
    pipeline.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred_proba = pipeline.predict_proba(X_valid)[:, 1]

    # Calculate Brier Score Loss for validation set
    brier_score = brier_score_loss(y_valid, np.clip(np.abs(y_pred_proba),0,1))
    print(f"Fold {fold} Brier Score Loss: {brier_score}")
    brier_scores.append(brier_score)

    # Accumulate test predictions
    test_predictions_fold = pipeline.predict_proba(test_data_transformed)[:, 1]
    test_predictions_accumulated += np.clip(np.abs(test_predictions_fold),0,1)

# Print mean Brier Score Loss for all folds
mean_brier_score = np.mean(brier_scores)
print(f"\nMean Brier Score Loss: {mean_brier_score}")

# Calculate mean of test predictions
test_predictions_mean = test_predictions_accumulated / stratified_kfold.n_splits
brier_score_test = brier_score_loss(y_test, np.clip(np.abs(test_predictions_mean),0,1))
print(f"Test Set Brier Score Loss: {brier_score_test}")

X_train = preprocessor.fit_transform(X)
test_data_transformed = preprocessor.transform(X_test)

# Create the final pipeline
pipeline = Pipeline(steps=[
    #https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
    ('feature_selection', VarianceThreshold(threshold=(.85 * (1 - .85)))),

    ('model', voting_model)])

# Fit the model to the training data
pipeline.fit(X_train, y)

test_predictions_fold = pipeline.predict_proba(test_data_transformed)[:, 1]

brier_score_test = brier_score_loss(y_test, np.clip(np.abs(test_predictions_fold),0,1))
print(f"Test Set Brier Score Loss: {brier_score_test}")
import pickle

with open("preprocessor.pickle","wb") as file:
    pickle.dump(preprocessor,file)

with open("pipeline.pickle","wb") as file:
    pickle.dump(pipeline,file)
