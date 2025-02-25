from scipy.stats import rankdata
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict


def calculate_adtm(results: Dict) -> Dict:
    adtm_scores = {}
    model_metrics = {}

    for dataset, result in results.items():
        for model_type, model_results in result.items():
            for model_name, metric in model_results.items():
                if model_name not in model_metrics:
                    model_metrics[model_name] = []
                model_metrics[model_name].append(metric)

    for model_name, metrics in model_metrics.items():
        adtm_sum = 0
        count = len(metrics)
        metric_best = max(metrics)
        metric_worst = min(metrics)
        for metric in metrics:
            adtm_sum += (metric - metric_worst) / (metric_best - metric_worst)
        adtm_scores[model_name] = adtm_sum / count if count > 0 else 0

    return adtm_scores


def calculate_ranks(results: Dict, highest_best: bool = True) -> Dict:

    model_scores = {}
    for dataset, data in results.items():
        for model_type, models in data.items():
            for model, score in models.items():
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(score)

    ranks = []
    for dataset, data in results.items():
        scores = []
        models = []
        for model_type, models_data in data.items():
            for model, score in models_data.items():
                scores.append(score)
                models.append(model)

        if not highest_best:
            scores = [-score for score in scores]
        ranks.append(rankdata(scores).tolist())
    ranks_df = pd.DataFrame(ranks, columns=models)
    average_ranks = {model: 0 for model in model_scores.keys()}
    for rank in ranks:
        for i, model in enumerate(models):
            average_ranks[model] += rank[i]
    average_ranks = {
        model: rank / len(results) for model, rank in average_ranks.items()
    }

    return average_ranks, ranks_df


def generate_CD_plot(average_ranks: Dict[str, float], ranks_df: pd.DataFrame):
    average_ranks_list = list(average_ranks.values())
    model_names = list(average_ranks.keys())

    # Check if average_ranks_list is not empty
    if len(average_ranks_list) == 0:
        print("Error: `average_ranks_list` is empty.")
        return

    # Perform the Nemenyi test
    nemenyi_results = sp.posthoc_nemenyi_friedman(ranks_df)

    # Generate the CD plot
    plt.figure(figsize=(16, 12))
    sp.sign_plot(nemenyi_results, labels=model_names)
    plt.show()
