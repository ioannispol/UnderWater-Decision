# xgb_train.py

import argparse
from pathlib import Path

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, learning_curve

# Import common functions from the classifier_base module
from underwater_decision.classifier_base import (
    load_dataset,
    encode_columns,
    get_features_target,
    split_dataset,
    prediction_model,
    prediction_accuracy,
    print_and_save_output,
    save_model,
)
from underwater_decision import MFLossObjective


# TODO: Use classes to encapsulate the functionality of the model
def train_xgboost_with_grid_search(features_train, target_train):
    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.1, 0.001],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.8, 1],
        "colsample_bytree": [0.8, 1],
    }
    xgb_model = xgb.XGBClassifier(
        use_label_encoder=False, eval_metric="auc", random_state=42, objective='multi: softmax'
    )
    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=8, scoring="accuracy", n_jobs=-1, verbose=3
    )
    grid_search.fit(features_train, target_train)
    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
    )


def train_xgboost(features_train, target_train, features_val, target_val, params):
    xgb_model = xgb.XGBClassifier(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42
    )
    
    eval_set = [(features_val, target_val)]
    xgb_model.fit(
        features_train,
        target_train,
        early_stopping_rounds=params['early_stopping_rounds'],
        eval_set=eval_set,
        verbose=True
    )
    
    return xgb_model



def plot_feature_importance(model, feature_columns, file_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(
        model, ax=ax, importance_type="weight", max_num_features=len(feature_columns)
    )
    plt.title("Feature Importance")
    plt.savefig(file_path)
    plt.close()


def plot_single_tree(model, tree_index, file_path):
    fig, ax = plt.subplots(figsize=(30, 30))
    xgb.plot_tree(model, num_trees=tree_index, ax=ax)
    plt.title(f"XGBoost Tree - Tree {tree_index}")
    plt.savefig(file_path)
    plt.close()


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
    file_path=None,
):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.legend(loc="best")
    if file_path:
        plt.savefig(file_path)
    plt.close()


def get_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate an XGBoost classifier."
    )
    parser.add_argument(
        "--data_file",
        "-d",
        type=str,
        default="default_synthetic_dataset1.csv",
        help="Filename of the dataset.",
    )
    parser.add_argument(
        "--model_file",
        "-m",
        type=str,
        default="xgboost_model_new.pkl",
        help="Filename to save the trained model.",
    )
    parser.add_argument(
        "--performance_file",
        "-p",
        type=str,
        default="xgboost_performance_new.txt",
        help="Filename for the model's performance metrics.",
    )
    parser.add_argument(
        "--feature_importance_file",
        "-fi",
        type=str,
        default="feature_importance_new.png",
        help="Filename for the feature importance plot.",
    )
    parser.add_argument(
        "--tree_plot_file",
        "-tp",
        type=str,
        default="xgb_tree_new.png",
        help="Filename for the XGBoost tree plot.",
    )
    parser.add_argument(
        "--learning_curve_file",
        "-lc",
        type=str,
        default="learning_curve_new.png",
        help="Filename for the learning curve plot.",
    )
    parser.add_argument(
        "--select_all",
        "-a",
        action="store_true",
        help="Run all analysis and generate all outputs.",
    )
    args = parser.parse_args()

    # Construct full file paths
    data_dir = Path("data/")
    args.data_file = data_dir / args.data_file
    args.model_file = data_dir / args.model_file
    args.performance_file = data_dir / args.performance_file
    args.feature_importance_file = data_dir / args.feature_importance_file
    args.tree_plot_file = data_dir / args.tree_plot_file
    args.learning_curve_file = data_dir / args.learning_curve_file

    return args


if __name__ == "__main__":
    args = get_args()

    if args.select_all:
        # Load the dataset
        dataset = load_dataset(str(args.data_file))

        # Encode the categorical columns
        dataset, encoders = encode_columns(
            dataset, ["platform", "item", "Recommended_Cleaning_Method"]
        )

        # Get the features and target
        feature_columns = [
            "platform",
            "year",
            "depthmin",
            "depthmax",
            "item",
            "hardPerc",
            "hardmm",
            "softPerc",
            "softmm",
            "Total_Area_Coverage",
        ]
        target_column = "Recommended_Cleaning_Method"
        features, target = get_features_target(dataset, feature_columns, target_column)

        # Split the dataset
        features_train, features_test, target_train, target_test = split_dataset(
            features, target, test_size=0.4
        )

        # Train the XGBoost model with Grid Search
        xgb_model, best_params, best_score = train_xgboost_with_grid_search(
            features_train, target_train
        )

        # Plot Feature Importance
        plot_feature_importance(
            xgb_model, feature_columns, str(args.feature_importance_file)
        )

        # Plot a single tree from the XGBoost model
        plot_single_tree(xgb_model, 0, str(args.tree_plot_file))

        # Plot the learning curve
        plot_learning_curve(
            xgb_model,
            "XGBoost Learning Curve",
            features_train,
            target_train,
            cv=5,
            n_jobs=-1,
            file_path=str(args.learning_curve_file),
        )

        # Predict the target and calculate accuracy
        target_pred = prediction_model(xgb_model, features_test, target_test)
        accuracy = prediction_accuracy(xgb_model, features_test, target_test)

        # Output results
        combined_output = f"Best parameters: {best_params}\nBest score from Grid Search: {best_score}\n{target_pred}\n\
            Accuracy: {accuracy}"
        print_and_save_output(combined_output, str(args.performance_file))

        # Save the XGBoost model
        save_model(xgb_model, str(args.model_file))
