# random_forest_train.py

import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Import common functionality from the classifier_base module
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


# TODO: Use classes to encapsulate the functionality of the model
def train_model(features_train, target_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(features_train, target_train)
    return model


def get_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Random Forest classifier."
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="default_synthetic_dataset.csv",
        help="Filename of the dataset.",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="random_forest_model.pkl",
        help="Filename to save the trained model.",
    )
    parser.add_argument(
        "--performance_file",
        type=str,
        default="random_forest_performance.txt",
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
    data_dir = Path("/workspaces/UnderWater-Decision/data/")
    args.data_file = data_dir / args.data_file
    args.model_file = data_dir / args.model_file
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

        # Train the Random Forest model
        model = train_model(features_train, target_train)

        # Predict the target and calculate accuracy
        target_pred = prediction_model(model, features_test, target_test)
        accuracy = prediction_accuracy(model, features_test, target_test)

        # Output results
        combined_output = f"{target_pred}\nAccuracy: {accuracy}"
        print_and_save_output(combined_output, str(args.performance_file))

        # Save the model
        save_model(model, str(args.model_file))
