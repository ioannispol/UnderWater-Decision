# decision_tree_train.py

import argparse
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

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
    get_decision_path,
)


class DecisionTreeModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.best_score = None

    def train(self, features_train, target_train):
        self.model = DecisionTreeClassifier(random_state=self.random_state)
        self.model.fit(features_train, target_train)
        return self.model

    def train_with_grid_search(self, features_train, target_train):
        param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        dtree = DecisionTreeClassifier(random_state=self.random_state)
        grid_search = GridSearchCV(
            dtree, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
        )
        grid_search.fit(features_train, target_train)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_

        return self.model, self.best_params, self.best_score


def plot_tree_diagram(model, feature_columns, class_encoder, file_path):
    plt.figure(figsize=(15, 10))
    plot_tree(
        model,
        filled=True,
        rounded=True,
        class_names=class_encoder.classes_,
        feature_names=feature_columns,
    )
    plt.savefig(file_path)
    plt.close()


def plot_correlation_matrix(df, file_path):
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix Heatmap")
    plt.savefig(file_path)
    plt.close()


def get_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Decision Tree classifier."
    )
    parser.add_argument(
        "--data_file",
        "-d",
        type=str,
        default="default_synthetic_dataset.csv",
        help="Filename of the dataset.",
    )
    parser.add_argument(
        "--model_file",
        "-m",
        type=str,
        default="decision_tree_model.pkl",
        help="Filename to save the trained model.",
    )
    parser.add_argument(
        "--tree_plot_file",
        "-t",
        type=str,
        default="decision_tree_model.png",
        help="Filename for the decision tree plot.",
    )
    parser.add_argument(
        "--correlation_plot_file",
        "-c",
        type=str,
        default="correlation_matrix.png",
        help="Filename for the correlation matrix plot.",
    )
    parser.add_argument(
        "--performance_file",
        "-p",
        type=str,
        default="model_performance.txt",
        help="Filename for the model's performance metrics.",
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
    args.tree_plot_file = data_dir / args.tree_plot_file
    args.correlation_plot_file = data_dir / args.correlation_plot_file
    args.performance_file = data_dir / args.performance_file

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
            features, target, test_size=0.3
        )

        # Train the Decision Tree model
        model_trainer = DecisionTreeModelTrainer()
        model, best_params, best_score = model_trainer.train_with_grid_search(
            features_train, target_train
        )

        # Print out the best parameters and best score from Grid Search
        print(f"Best parameters: {best_params}")
        print(f"Best score from Grid Search: {best_score}")

        # Get decision path for a specific sample
        sample_id = 0  # You can change this to any valid index of a sample in X_test
        decision_path_output = get_decision_path(model, features_test, sample_id)

        # Predict the target and calculate accuracy
        target_pred = prediction_model(model, features_test, target_test)
        accuracy = prediction_accuracy(model, features_test, target_test)

        # Output results
        combined_output = (
            f"{target_pred}\nAccuracy: {accuracy}\n\n{decision_path_output}"
        )
        print_and_save_output(combined_output, str(args.performance_file))

        # Visualize and save the Decision Tree
        plot_tree_diagram(
            model, feature_columns, encoders[target_column], str(args.tree_plot_file)
        )
        plot_correlation_matrix(dataset, str(args.correlation_plot_file))

        # Save the Decision Tree model
        save_model(model, str(args.model_file))
