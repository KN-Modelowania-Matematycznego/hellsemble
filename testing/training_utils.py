import os
import pandas as pd
from sklearn.metrics import accuracy_score
from typing import Dict
from hellsemble.hellsemble import Hellsemble
from hellsemble.estimator_generator import EstimatorGenerator
from hellsemble.predction_generator import FixedThresholdPredictionGenerator
from sklearn.linear_model import LogisticRegression
import json

def train_and_test_models(train_file: str, test_file: str, models: Dict[str, object], n_runs: int = 100) -> Dict[str, float]:
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    
    results = {}
    
    for name, model in models.items():

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

    return results

def run_and_save_base_models(train_dir: str, test_dir: str, output_dir: str, models: Dict[str, object], n_runs: int = 100):
    results = {}

    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]

    for train_file, test_file in zip(train_files, test_files):
        dataset_name = os.path.basename(train_file).replace('.csv', '')
        results[dataset_name] = train_and_test_models(train_file, test_file, models = models)

    with open(f'{output_dir}/all_results.json', 'w') as json_file:
        json.dump(results, json_file)

def train_and_test_hellsemble(train_file: str, test_file: str, estimators_generator: EstimatorGenerator, models: list, mode: str = 'Sequential',n_runs = 100):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    estimator = Hellsemble(
        estimators_generator(
            models
        ),
        FixedThresholdPredictionGenerator(0.5),
        LogisticRegression(),
        mode = mode
    )
    

    model = estimator.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

def run_and_save_hellsemble(train_dir: str, test_dir: str, output_dir: str, estimators_generator: EstimatorGenerator, models: list, mode: 'str' = 'Sequential',n_runs: int = 100):
    results = {}

    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]

    for train_file, test_file in zip(train_files, test_files):
        dataset_name = os.path.basename(train_file).replace('.csv', '')
        try:
            results[dataset_name] = train_and_test_hellsemble(train_file, test_file, estimators_generator, models, mode)
        except Exception as e:
            print(f'Error in dataset {dataset_name}: {e}')

    with open(f'{output_dir}/hellsemble_results_{mode}.json', 'w') as json_file:
        json.dump(results, json_file)