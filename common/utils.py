from sklearn.model_selection import train_test_split
import pandas as pd
import json
def add_features(df):
    # Family Size
    df['is_elderly'] = (df['age'] >= 60).astype(int)
    df['has_housing_loan'] = (df['housing'] == 'yes').astype(int)
    df['is_married'] = (df['marital'] == 'married').astype(int)
    df['has_previous_contact'] = (df['contact'] != 'unknown').astype(int)
    df['is_tertiary_educated'] = (df['education'] == 'tertiary').astype(int)
    df['is_admin_job'] = (df['job'] == 'admin.').astype(int)
    df['has_default'] = (df['default'] == 'yes').astype(int)
    df['is_month_may'] = (df['month'] == 'may').astype(int)
    df['is_loan'] = (df['loan'] == 'yes').astype(int)
    df['campaign_greater_than_1'] = (df['campaign'] > 1).astype(int)

    return df
def split_data(df):
    X, X_test, y, y_test = train_test_split(
        df.drop("deposit", axis=1), df["deposit"], test_size=0.2, stratify=df["deposit"], random_state=42)
    # Separate features and target variable
    X = df.drop("deposit", axis=1)
    y = df["deposit"]
    y = y.apply(lambda x :1 if x=="yes" else 0)
    y_test = y_test.apply(lambda x :1 if x=="yes" else 0)
    return X, X_test, y, y_test

def read_data(artifact_dir):
    """
    return train
    """
    dtypes = {'age': 'int64',
                 'job': 'O',
                 'marital': 'O',
                 'education': 'O',
                 'default': 'O',
                 'balance': 'int64',
                 'housing': 'O',
                 'loan': 'O',
                 'contact': 'O',
                 'day': 'int64',
                 'month': 'O',
                 'duration': 'int64',
                 'campaign': 'int64',
                 'pdays': 'int64',
                 'previous': 'int64',
                 'poutcome': 'O',
                 'deposit':'int64'
                 }
    with open(f"{artifact_dir}") as json_data:
        data = json.load(json_data)
        train = pd.DataFrame(data = data["data"],columns=data["columns"])
        json_data.close()

    for i in dtypes:
        try:
            train[i] = train[i].astype(dtypes[i])
        except:pass
    return train