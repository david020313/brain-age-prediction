"""
Machine Learning Ensemble: XGBoost on CNN3D feature outputs.

Extracts feature predictions from pre-trained CNN3D models (C1/C2), then
uses XGBoost regression for the final brain age prediction.

Usage:
    python -m combine.ml_ensemble --train_csv train.csv --test_csv test.csv
"""

import argparse
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    parser = argparse.ArgumentParser(description="XGBoost ensemble for brain age")
    parser.add_argument("--train_csv", type=str, required=True,
                        help="Training CSV with columns: ID, Brain_Age, features...")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="Test CSV with same format")
    parser.add_argument("--save_model", type=str, default="xgboost_model.json")
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--n_estimators", type=int, default=300)
    args = parser.parse_args()

    # Load data
    train_data = pd.read_csv(args.train_csv)
    test_data = pd.read_csv(args.test_csv)

    X_train = train_data.drop(columns=['ID', 'Brain_Age'])
    y_train = train_data['Brain_Age']
    X_test = test_data.drop(columns=['ID', 'Brain_Age'])
    y_test = test_data['Brain_Age']

    # XGBoost training
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'objective': 'reg:squarederror',
        'tree_method': 'auto',
    }

    print("Training XGBoost ...")
    t0 = time.time()
    model = xgb.train(params, dtrain, num_boost_round=args.n_estimators,
                      verbose_eval=50)
    print(f"Training took {time.time() - t0:.1f}s")

    # Evaluate
    y_pred = model.predict(dtest)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nResults: MAE={mae:.4f}, MSE={mse:.4f}")

    # Save
    model.save_model(args.save_model)
    print(f"Model saved to {args.save_model}")


if __name__ == "__main__":
    main()
