# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

import mlflow
import mlflow.sklearn


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser("train")

    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    # This should map to the component/pipeline output 'model_output'
    parser.add_argument("--model_output", type=str, help="Path of output model")

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=50,
        help="Number of trees for the RandomForestRegressor",
    )

    parser.add_argument(
        "--criterion",
        type=str,
        default="mse",  # use a valid regressor criterion
        help="The function to measure the quality of a split",
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help=(
            "The maximum depth of the tree. If None, then nodes are expanded until "
            "all the leaves contain less than min_samples_split samples."
        ),
    )

    args = parser.parse_args()
    return args


def main(args):
    """Read train and test datasets, train model, evaluate model, save trained model"""

    # Read train and test data from CSV
    train_df = pd.read_csv(Path(args.train_data) / "train.csv")
    test_df = pd.read_csv(Path(args.test_data) / "test.csv")

    # Split into features and target
    y_train = train_df["price"]
    X_train = train_df.drop(columns=["price"])

    y_test = test_df["price"]
    X_test = test_df.drop(columns=["price"])

    # Initialize and train model
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        criterion=args.criterion,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Log hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("criterion", args.criterion)

    # Predict and evaluate
    yhat_test = model.predict(X_test)
    mse = mean_squared_error(y_test, yhat_test)
    print("Mean Squared Error of RandomForest Regressor on test set: {:.2f}".format(mse))
    mlflow.log_metric("mse", float(mse))

    # Ensure output dir exists and save a pickle for Azure ML to pick up
    model_output_dir = Path(args.model_output)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_dir / "model.pkl")

    # Also log the model to MLflow under the same artifact path name
    # artifact_path is a *name* inside the MLflow run, not a full filesystem path
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")


if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}",
        f"Criterion: {args.criterion}",
    ]

    for line in lines:
        print(line)

    main(args)
    mlflow.end_run()
