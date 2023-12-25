import pandas as pd
import numpy as np
import os , json
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
import wandb
from wandb.sklearn import plot_precision_recall, plot_feature_importances
from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc

name_="mustafakeser"
project_="marketing-campaign-wb"
entity_=None

run = wandb.init(
                project=project_, 
                entity=entity_, 
                   job_type="split",
                name = "04-Train",
                tags = ["SPLIT"]
                
    )

if "artifacts" not in os.listdir():
    raw_data_at = run.use_artifact('mustafakeser/marketing-campaign-wb/marketing-campaign-wb:v1', 
                                                    type='raw_data')
    artifact_dir = raw_data_at.download()

    df = read_data(os.path.join(artifact_dir,"df.table.json"))
else: 
    df = read_data(os.path.join(os.getcwd(),"artifacts/marketing-campaign-wb:v1/df.table.json"))




X, X_test, y, y_test = split_data(df)

split_data_at = wandb.Artifact("marketing-campaign-wb-split-dataset", type="split_data")
df_train = pd.concat([X,y],axis=1)
df_val = pd.concat([X_test,y_test],axis=1)
tbl_df_train = wandb.Table(data=df_train)
tbl_df_val   = wandb.Table(data=df_val)
wandb.log({"train_df": tbl_df_train})
wandb.log({"test_df": tbl_df_val})
split_data_at.add(tbl_df_train, "df_train")
split_data_at.add(tbl_df_val, "df_val")
run.log_artifact(split_data_at)
run.finish()


run = wandb.init(
                project=project_, 
                entity=entity_, 
                   job_type="train",
                name = "04-Train",
                tags = ["TRAIN"]
                
    )

if "artifacts" not in os.listdir("artifacts"):
    raw_data_at = run.use_artifact('mustafakeser/marketing-campaign-wb/marketing-campaign-wb-split-dataset:v0', 
                                                    type='split_data')
    artifact_dir = raw_data_at.download()
    X = read_data(os.path.join(artifact_dir,"df_train.table.json"))
else: 
    X = read_data(os.path.join(os.getcwd(),"artifacts/marketing-campaign-wb-split-dataset:v0/df_train.table.json"))

if "artifacts" not in os.listdir("artifacts"):
    raw_data_at = run.use_artifact('mustafakeser/marketing-campaign-wb/marketing-campaign-wb-split-dataset:v0', 
                                                    type='split_data')
    artifact_dir = raw_data_at.download()
    X_test = read_data(os.path.join(artifact_dir,"df_val.table.json"))

else: 
    X_test = read_data(os.path.join(os.getcwd(),"artifacts/marketing-campaign-wb-split-dataset:v0/df_val.table.json"))


X = X.drop(columns=["deposit"])
X_test = X_test.drop(columns=["deposit"])

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
    y_pred_probas = pipeline.predict_proba(X_valid)
    
    y_pred = pipeline.predict(X_valid)
    # Calculate Brier Score Loss for validation set
    brier_score = brier_score_loss(y_valid, np.clip(np.abs(y_pred_proba),0,1))
    wandb.log({f"brier_score_fold_{fold}": brier_score})

    print(f"Fold {fold} Brier Score Loss: {brier_score}")
    brier_scores.append(brier_score)

    # Accumulate test predictions
    test_predictions_fold = pipeline.predict_proba(test_data_transformed)[:, 1]
    test_predictions_accumulated += np.clip(np.abs(test_predictions_fold),0,1)

    #report to wandb
    plot_roc(y_valid, y_pred_probas, ["no-deposit","deposit"])
    plot_precision_recall(y_valid, y_pred_probas, ["no-deposit","deposit"])
    plot_class_proportions(y_train, y_valid, ["no-deposit","deposit"])
    plot_learning_curve(pipeline, X_train, y_train)
    plot_learning_curve(pipeline, X_valid, y_valid)

    fold_data_at = wandb.Artifact(f"marketing-campaign-wb-fold-dataset-{fold}", type="fold_data")
    df_train = pd.concat([pd.DataFrame(X_train,columns=[f"col_{i}" for i in range(61)]),pd.DataFrame(y_train.reset_index(drop=True),columns=["deposit"])],axis=1)
    df_val = pd.concat([pd.DataFrame(X_valid,columns=[f"col_{i}" for i in range(61)]),pd.DataFrame(y_valid.reset_index(drop=True),columns=["deposit"])],axis=1)
    tbl_df_train = wandb.Table(data=df_train)
    tbl_df_val   = wandb.Table(data=df_val)
    wandb.log({f"fold_{fold}_train": tbl_df_train})
    wandb.log({f"fold_{fold}_val": tbl_df_val})
    fold_data_at.add(tbl_df_train, f"fold_{fold}_train")
    fold_data_at.add(tbl_df_val, f"fold_{fold}_val")
    run.log_artifact(fold_data_at)
    
    wandb.sklearn.plot_classifier(
    pipeline,
    X_train,
    X_valid,
    y_train,
    y_valid,
    y_pred,
    y_pred_probas,
    ["no-deposit","deposit"],
    model_name=f"pipeline_fold_{fold}",
    feature_names=None,
    )



# Print mean Brier Score Loss for all folds
mean_brier_score = np.mean(brier_scores)
print(f"\nMean Brier Score Loss: {mean_brier_score}")
wandb.log({f"Mean_Val_brier_score": mean_brier_score})

# Calculate mean of test predictions
test_predictions_mean = test_predictions_accumulated / stratified_kfold.n_splits
brier_score_test = brier_score_loss(y_test, np.clip(np.abs(test_predictions_mean),0,1))
print(f"Test Set Brier Score Loss: {brier_score_test}")
wandb.log({f"Test_Set_brier_score": brier_score_test})


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
wandb.log({f"Allfit_Test_Set_brier_score": brier_score_test})



import pickle

with open("preprocessor.pickle","wb") as file:
    pickle.dump(preprocessor,file)

with open("pipeline.pickle","wb") as file:
    pickle.dump(pipeline,file)
    
wandb.finish()