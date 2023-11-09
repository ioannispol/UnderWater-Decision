# compare_models.py

import argparse

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Import common functions from the classifier_base module
from classifier_base import (load_dataset, encode_columns, get_features_target, split_dataset)


# Function to train Decision Tree
def train_decision_tree(features_train, target_train):
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(features_train, target_train)
    return dt_model


# Function to train Random Forest
def train_random_forest(features_train, target_train):
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(features_train, target_train)
    return rf_model


# Function to train XGBoost
def train_xgboost(features_train, target_train):
    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(features_train, target_train)
    return xgb_model


# Function to evaluate models
def evaluate_model(model, features_test, target_test):
    predictions = model.predict(features_test)
    accuracy = accuracy_score(target_test, predictions)
    report = classification_report(target_test, predictions)
    return accuracy, report


def main():
    parser = argparse.ArgumentParser(description="Compare Decision Trees, Random Forests, and XGBoost models.")
    parser.add_argument("--data_file", type=str, default="default_synthetic_dataset.csv",
                        help="Filename of the dataset.")
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(args.data_file)

    # Encode the categorical columns
    # Assuming encode_columns returns a tuple: (encoded_dataset, encoders)
    encoded_dataset, encoders = encode_columns(dataset, ['platform', 'item', 'Recommended_Cleaning_Method'])

    # Get the features and target
    feature_columns = ['platform', 'year', 'depthmin', 'depthmax', 'item', 'hardPerc', 'hardmm', 'softPerc', 'softmm',
                       'Total_Area_Coverage']
    target_column = 'Recommended_Cleaning_Method'
    features, target = get_features_target(encoded_dataset, feature_columns, target_column)

    # Split the dataset
    features_train, features_test, target_train, target_test = split_dataset(features, target, test_size=0.3)

    # Train and evaluate Decision Tree
    dt_model = train_decision_tree(features_train, target_train)
    dt_accuracy, dt_report = evaluate_model(dt_model, features_test, target_test)

    # Train and evaluate Random Forest
    rf_model = train_random_forest(features_train, target_train)
    rf_accuracy, rf_report = evaluate_model(rf_model, features_test, target_test)

    # Train and evaluate XGBoost
    xgb_model = train_xgboost(features_train, target_train)
    xgb_accuracy, xgb_report = evaluate_model(xgb_model, features_test, target_test)

    # Print the comparison
    print("Decision Tree Accuracy: ", dt_accuracy)
    print("Random Forest Accuracy: ", rf_accuracy)
    print("XGBoost Accuracy: ", xgb_accuracy)

    print("\nDecision Tree Classification Report:\n", dt_report)
    print("\nRandom Forest Classification Report:\n", rf_report)
    print("\nXGBoost Classification Report:\n", xgb_report)


if __name__ == "__main__":
    main()
