import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from underwater_decision.mf_loss import MFLossObjective


def load_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    # Encode categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = LabelEncoder().fit_transform(data[col])
    X = data.drop('Recommended_Cleaning_Method', axis=1)  # Replace 'target_column_name' with your actual target column
    y = data['Recommended_Cleaning_Method']  # Actual target column name
    return X, y

def calculate_class_weights(y, smooth_factor=0.5):
    """
    Calculate class weights inversely proportional to class frequencies.
    """
    counter = np.bincount(y)
    if smooth_factor > 0:
        counter = counter + smooth_factor
    weights = 1. / counter
    normalized_weights = weights / weights.sum()
    return normalized_weights.tolist()

def train_xgboost_with_custom_loss(X_train, y_train, X_val, y_val, weights, lambda_reg):
    # Ensure 'enable_categorical' is not needed by converting all features to numeric
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    custom_loss_obj = MFLossObjective(weights, lambda_reg)
    params = {
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 1,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        # Additional parameters as needed
    }

    bst = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtrain, 'train'), (dval, 'eval')],
                    early_stopping_rounds=10, obj=custom_loss_obj.xgb_obj)

    return bst


if __name__=="__main__":
    filepath ='/workspaces/UnderWater-Decision/data/default_synthetic_dataset1.csv'
    X, y = load_preprocess_data(filepath)
    weights = calculate_class_weights(y)  # Adjust this function as per your need
    lambda_reg = 0.5

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    bst = train_xgboost_with_custom_loss(X_train, y_train, X_val, y_val, weights, lambda_reg)
