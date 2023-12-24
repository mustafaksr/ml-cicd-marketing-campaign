# Number of folds
n_folds = 3
seed = 42 
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import pandas as pd
import numpy as np
import os, json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import  VarianceThreshold
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.ensemble import (RandomForestClassifier, 
                                GradientBoostingClassifier, 
                                VotingClassifier, 
                                BaggingClassifier ,
                                )
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict ,train_test_split
import warnings
from utils import *
warnings.filterwarnings("ignore")
df = pd.read_csv(os.path.join(os.getcwd(),"../train/bank.csv")).sample(frac=0.5, random_state=42 ).reset_index(drop=True)
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

models=[
    ("bag",SGDClassifier()),
    ("bag2",LogisticRegression()),
    ('gb', GradientBoostingClassifier()),
    ('xgb', XGBClassifier()),
    ('lgb', LGBMClassifier()),
    ("log", LogisticRegression()),
    ("sgd", SGDClassifier()),
   ('rf', RandomForestClassifier()),
    ('cat', CatBoostClassifier()),
]

kf = StratifiedKFold(n_splits=n_folds,shuffle=True, random_state=seed)

for i,(train_idx, test_idx) in enumerate(kf.split(X,y)):
# for i, (train_idx, test_idx) in enumerate(stratified_group_kfold.split(X, y, groups=groups)):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    break

tune = True
# Define the parameter distributions for Gaussian Naive Bayes
bag_params = {
    # Define hyperparameter distributions and their possible values here
    'model__max_iter': randint(6000, 8000),
    'model__tol': uniform(1e-5, 1e-4)

}

bag2_params = {
    # Define hyperparameter distributions and their possible values here
    'model__solver': ['sag','lbfgs']
}

gb_params = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': randint(3, 10),
    'model__learning_rate': uniform(1e-03, 3e-01),
    'model__min_samples_leaf': randint(50, 200),
    
    
}

xgb_params = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': randint(3, 10),
    'model__learning_rate': uniform(1e-03, 3e-01),
    'model__min_child_weight': randint(50, 200),
    
    
}

lgb_params = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': randint(3, 10),
    'model__learning_rate': uniform(1e-03, 3e-01),
    'model__min_data_in_leaf': randint(15,40),
    'model__boosting_type':  ["gbdt"]
    
    
}

log_params = {
    # Define hyperparameter distributions and their possible values here
    'model__solver': ['sag','lbfgs']
}


sgd_params = {
    'model__max_iter': randint(5000, 8000),
    'model__tol': uniform(1e-5, 1e-4)

}


rf_params = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': randint(3, 10),
    'model__min_samples_leaf': randint(50, 200),  
}

cat_params = {
    'model__max_bin': randint(75, 150),
    'model__depth': randint(3, 10),
    'model__learning_rate':uniform(1e-3, 1e-2),  
    'model__one_hot_max_size': randint(8, 16),
    
}



# Create a dictionary to map model names to their respective parameter distributions
param_distributions = {
        
    'bag': bag_params,
    'bag2': bag2_params,
    'gb': gb_params, 
    'xgb': xgb_params,
    'lgb': lgb_params, 
    'log': log_params, 
    'sgd': sgd_params, 
    'rf': rf_params, 
    'cat': cat_params, 

}

if tune:
    # Loop through each model and perform RandomizedSearchCV
    best_models = []
    best_scores = {}
    best_params = {}

    for model_name, model in models:
        # Create a pipeline for the current model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Get the parameter distributions for the current model
        param_dist = param_distributions.get(model_name, {})

        # Perform RandomizedSearchCV for the current model
        randomized_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=1,  # Number of random parameter settings to sample
            scoring='roc_auc',  # Choose an appropriate scoring metric
            cv=2,  # Number of cross-validation folds
            n_jobs=-1,  # Use all available CPU cores
            verbose=2  # Increase verbosity for detailed output
        )

        # Fit the RandomizedSearchCV to your data
        randomized_search.fit(X_train, y_train)

        # Get the best model and its parameters
        best_model = randomized_search.best_estimator_
        best_param = {key.replace('model__', ''): value for key, value in randomized_search.best_params_.items()}
        # Store the best model for the current model
    #     best_models[model_name] = (best_model, best_params)
        best_models.append((model_name,best_model.named_steps["model"]))
        best_scores[model_name] = randomized_search.best_score_
        best_params[model_name] = best_param
        print(f"\n\nmodelname: {model_name}, cv score : {round(randomized_search.best_score_,5)}\n\n")

    # Now you have the best-tuned models for each algorithm in best_models dictionary
    print("\nbest scores",best_scores)
    print("\nbest parameters",best_params)
    
with open("best_parameters.json","w") as file: 
    json.dump(best_params,file)
