import json
import os
import re
import csv

def load_data(path):
    with open(path) as f:
        data = json.load(f)
    return data

def process_string(s: str) -> list[str]:
    s = re.findall(r"\b[a-zA-Z][\w_]*\([^)]*\)", s)
    s = [fact.replace(" ", "") for fact in s]
    return s

def save_results(results, file_path='llama_results.json'):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as outfile:
        outfile.write(json.dumps(results))

def evaluate_predictions(y_pred, y_true):
    # Convert x and y lists to sets
    y_pred, y_true = set(y_pred), set(y_true)

    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    true_positives = len(y_pred.intersection(y_true))
    false_positives = len(y_pred - y_true)
    false_negatives = len(y_true - y_pred)

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = true_positives / len(y_pred) if len(y_pred) > 0 else 0

    # Output results
    results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    return results


def evaluate_model(model_name, data, stats_file):
    s = []
    overall = {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1 Score': 0.0}
    unique_problems = set([problem['problem_name'] for problem in data])
    problem_metrics = {}
    for problem in unique_problems:
        problem_metrics[problem] = {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1 Score': 0.0, 'Count': 0}
    
    for instance in data:
        y_true, y_pred = instance["output"], instance[model_name]
        y_true = process_string(y_true)
        y_pred = process_string(y_pred)
        instance_metrics = evaluate_predictions(y_pred, y_true)
        
        for metric, value in instance_metrics.items():
            overall[metric] += value
            problem_metrics[instance['problem_name']][metric] += value
        problem_metrics[instance['problem_name']]['Count'] += 1

    table = [['Problem Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']]
    for problem, metrics in problem_metrics.items():
        s.append([problem] + [f'{value/metrics["Count"]:.2f}' for metric, value in metrics.items() if metric != 'Count'])

    s = sorted(s, key=lambda x: x[0])
    s.append(["Overall Results"] + [f'{value/len(data):.2f}' if len(data) > 0 else 0 for _, value in overall.items()])
    table.extend(s)

    if stats_file is not None:
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        with open(stats_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(table)
    else:
        return table