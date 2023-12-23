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
