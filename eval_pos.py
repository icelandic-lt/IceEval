import json
import os
import argparse
from typing import List, Tuple
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


def load_json(file_path: str) -> List[dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_predictions(file_path: str) -> List[List[str]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip().split() for line in f]

def calculate_accuracy(gold_tags: List[str], pred_tags: List[str], ignore_tags: List[str]) -> Tuple[int, int]:
    correct = sum(1 for g, p in zip(gold_tags, pred_tags) if g == p and g not in ignore_tags)
    total = sum(1 for g in gold_tags if g not in ignore_tags)
    return correct, total


def get_top_n_incorrect_predictions(gold_tags, pred_tags, top_n):
    # Count all incorrect predictions
    incorrect_predictions = Counter()
    for gold, pred in zip(gold_tags, pred_tags):
        if gold != pred:
            incorrect_predictions[(gold, pred)] += 1

    # Sort by frequency in descending order
    sorted_predictions = sorted(incorrect_predictions.items(), key=lambda x: x[1], reverse=True)
    return sorted_predictions[:top_n] if top_n is not None else sorted_predictions


def save_top_confusions(gold_tags, pred_tags, top_n, output_file):
    top_confusions = get_top_n_incorrect_predictions(gold_tags, pred_tags, top_n)

    with open(output_file, 'w') as f:
        f.write("True Label\tPredicted Label\tCount\n")
        for (true_label, pred_label), count in top_confusions:
            f.write(f"{true_label}\t{pred_label}\t{count}\n")


def evaluate_fold(test_file: str, pred_file: str, ignore_tags: List[str], full: bool, fold: int, output_dir: str, plot_cm: bool) -> Tuple[float, int, bool, List[str], List[str]]:
    gold_data = load_json(test_file)
    pred_data = load_predictions(pred_file)

    total_correct = 0
    total_tokens = 0
    discrepancy_found = False
    all_gold_tags = []
    all_pred_tags = []

    for gold, pred in zip(gold_data, pred_data):
        gold_tags = gold['tags']
        if len(gold_tags) != len(pred):
            discrepancy_found = True

        if full:
            eval_gold_tags = gold_tags
            eval_pred_tags = pred
        else:
            eval_gold_tags = gold_tags[:len(pred)]
            eval_pred_tags = pred

        correct, total = calculate_accuracy(eval_gold_tags, eval_pred_tags, ignore_tags)
        total_correct += correct
        total_tokens += total

        for g, p in zip(eval_gold_tags, eval_pred_tags):
            if g not in ignore_tags:
                all_gold_tags.append(g)
                all_pred_tags.append(p)

    fold_accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    top_confusions_file = os.path.join(output_dir, f'top_confusions_fold_{fold}.tsv')
    save_top_confusions(all_gold_tags, all_pred_tags, 50, top_confusions_file)

    if plot_cm:
        cm_file = os.path.join(output_dir, f'confusion_matrix_fold_{fold}.png')
        create_confusion_matrix(all_gold_tags, all_pred_tags, 30, cm_file)

    return fold_accuracy, total_tokens, discrepancy_found, all_gold_tags, all_pred_tags


def plot_confusion_matrix(cm, classes, title, output_file):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def create_confusion_matrix(gold_tags, pred_tags, top_n, output_file):
    top_confusions = get_top_n_incorrect_predictions(gold_tags, pred_tags, top_n)

    # Get unique labels from top confusions
    unique_labels = sorted(set(gold for (gold, _), _ in top_confusions) |
                           set(pred for (_, pred), _ in top_confusions))

    # Create confusion matrix
    cm = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    for (true, pred), count in top_confusions:
        true_index = label_to_index[true]
        pred_index = label_to_index[pred]
        cm[true_index][pred_index] = count

    # Plot the confusion matrix
    plot_confusion_matrix(cm, unique_labels, f'Top {top_n} Misclassifications', output_file)


def main(splits_dir: str, results_dir: str, ignore_tags: List[str], full: bool, output_dir: str, plot_cm: bool):
    total_correct = 0
    total_tokens = 0
    all_gold_tags = []
    all_pred_tags = []

    for fold in range(1, 11):
        test_file = os.path.join(splits_dir, f'{fold:02d}-test.json')
        pred_file = os.path.join(results_dir, f'{fold:02d}', 'predictions.txt')

        print(f"\nProcessing Fold {fold:02d}")
        fold_accuracy, fold_tokens, discrepancy, fold_gold_tags, fold_pred_tags = evaluate_fold(test_file, pred_file, ignore_tags, full, fold, output_dir, plot_cm)
        total_correct += fold_accuracy * fold_tokens
        total_tokens += fold_tokens
        all_gold_tags.extend(fold_gold_tags)
        all_pred_tags.extend(fold_pred_tags)

        print(f"Fold {fold:02d} Accuracy: {fold_accuracy:.4f} (Tokens: {fold_tokens})")
        if discrepancy:
            print(f"Warning: Discrepancy found in Fold {fold:02d}. Some sequences have different lengths in gold and prediction data.")

    overall_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    print(f"\nOverall Accuracy: {overall_accuracy:.4f} (Total Tokens: {total_tokens})")

    top_confusions_file = os.path.join(output_dir, 'top_confusions_overall.tsv')
    save_top_confusions(all_gold_tags, all_pred_tags, 50, top_confusions_file)

    if plot_cm:
        cm_file = os.path.join(output_dir, 'confusion_matrix_overall.png')
        create_confusion_matrix(all_gold_tags, all_pred_tags, 30, cm_file)

    if not full:
        print("Note: Evaluation was performed on truncated sequences matching prediction lengths.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate POS tagging accuracy across folds")
    parser.add_argument("--splits_dir", required=True, help="Directory containing the fold splits for training")
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing the results (parent directory of fold results)")
    parser.add_argument("--ignore_tags", nargs="+", default=[], help="List of tags to ignore in evaluation")
    parser.add_argument("--full", action="store_true", help="Evaluate full sequences, including untagged tokens")
    parser.add_argument("--output_dir", default=".", help="Directory to save output files")
    parser.add_argument("--confusion_matrix", action="store_true", help="Generate confusion matrix plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args.splits_dir, args.results_dir, args.ignore_tags, args.full, args.output_dir, args.confusion_matrix)
