import argparse
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parent.parent


def parse_args():
	parser = argparse.ArgumentParser(description="Train and track a classification model")
	parser.add_argument("--params-file", type=Path, default=ROOT_DIR / "params.yaml")
	parser.add_argument("--model-type", choices=["random_forest", "xgboost", "both"])
	parser.add_argument("--n-estimators", type=int)
	parser.add_argument("--max-depth", type=int)
	parser.add_argument("--random-state", type=int)
	parser.add_argument("--learning-rate", type=float)
	parser.add_argument("--subsample", type=float)
	parser.add_argument("--colsample-bytree", type=float)
	return parser.parse_args()


def load_params(args):
	with args.params_file.open() as params_file:
		configured = yaml.safe_load(params_file).get("model", {})

	params = {
		"model_type": configured.get("model_type", "random_forest"),
		"n_estimators": configured.get("n_estimators", 100),
		"max_depth": configured.get("max_depth"),
		"random_state": configured.get("random_state", 42),
		"learning_rate": configured.get("learning_rate", 0.1),
		"subsample": configured.get("subsample", 1.0),
		"colsample_bytree": configured.get("colsample_bytree", 1.0),
	}
	for name in params:
		override = getattr(args, name)
		if override is not None:
			params[name] = override
	return params


def build_model(params):
	model_type = params["model_type"]
	if model_type == "random_forest":
		return RandomForestClassifier(
			n_estimators=params["n_estimators"],
			max_depth=params["max_depth"],
			random_state=params["random_state"],
		)
	if model_type == "xgboost":
		return XGBClassifier(
			n_estimators=params["n_estimators"],
			max_depth=params["max_depth"] or 6,
			learning_rate=params["learning_rate"],
			subsample=params["subsample"],
			colsample_bytree=params["colsample_bytree"],
			random_state=params["random_state"],
			eval_metric="logloss",
			n_jobs=-1,
		)
	raise ValueError(f"Unsupported model type: {model_type}")


def main():
	args = parse_args()
	params = load_params(args)
	tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file:{ROOT_DIR / 'mlruns'}")
	mlflow.set_tracking_uri(tracking_uri)
	mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "water-potability"))

	train_data = pd.read_csv(ROOT_DIR / "data/processed/train_processed.csv")
	test_data = pd.read_csv(ROOT_DIR / "data/processed/test_processed.csv")
	X_train = train_data.drop(columns=["Potability"])
	y_train = train_data["Potability"].values
	X_test = test_data.drop(columns=["Potability"])
	y_test = test_data["Potability"].values
	model_types = ["random_forest", "xgboost"] if params["model_type"] == "both" else [params["model_type"]]
	best_model = None
	best_accuracy = -1.0
	best_run_id = None

	for model_type in model_types:
		trial_params = {**params, "model_type": model_type}
		with mlflow.start_run() as run:
			clf = build_model(trial_params)
			clf.fit(X_train, y_train)
			predictions = clf.predict(X_train)
			test_predictions = clf.predict(X_test)
			test_accuracy = accuracy_score(y_test, test_predictions)
			mlflow.log_params(trial_params)
			mlflow.log_metric("train_accuracy", accuracy_score(y_train, predictions))
			mlflow.log_metric("train_f1_score", f1_score(y_train, predictions, zero_division=0))
			mlflow.log_metric("test_accuracy", test_accuracy)
			mlflow.log_metric("test_f1_score", f1_score(y_test, test_predictions, zero_division=0))
			mlflow.sklearn.log_model(clf, "model")
			if test_accuracy > best_accuracy:
				best_model = clf
				best_accuracy = test_accuracy
				best_run_id = run.info.run_id

	(ROOT_DIR / ".mlflow_run_id").write_text(best_run_id)

	with (ROOT_DIR / "model.pkl").open("wb") as model_file:
		pickle.dump(best_model, model_file)


if __name__ == "__main__":
	main()