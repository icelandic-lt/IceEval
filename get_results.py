#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import itertools
import json
import os
import re


def get_transformer_results(run_dir, metric):
    """Get the results for token classification tasks (NER, POS) which were evaluated using the Transformer library.

    :param run_dir: The directory containing the results.
    :param metric: The metric which should be returned.
    :return: The score obtained for this run.
    """
    results_path = os.path.join(run_dir, 'predict_results.json')

    if os.path.exists(results_path):
        with open(results_path, encoding='utf-8') as f:
            doc = json.load(f)

        return doc[metric]


def get_transformersum_results(run_dir, metric):
    """Get the results for the text summarization task which was evaluated using the TransformerSum library.

    :param run_dir: The directory containing the results.
    :param metric: The metric which should be returned.
    :return: The score obtained for this run.
    """
    results_path = os.path.join(run_dir, 'results.jsonl')
    with open(results_path, encoding='utf-8') as f:
        all_scores = [json.loads(line.strip())[metric] for line in f]

    return sum(all_scores) / len(all_scores)


def get_diaparser_results(run_dir, metric):
    """Get the results for the dependency parsing task which was evaluated using the DiaParser library.

    :param run_dir: The directory containing the results.
    :param metric: The metric which should be returned.
    :return: The score obtained for this run.
    """
    results = {}

    train_log = [name for name in os.listdir(run_dir) if name.endswith('.train.log')]

    if not train_log:
        return results

    log_name = train_log[-1]
    log_path = os.path.join(run_dir, log_name)

    with open(log_path, encoding='utf-8') as f:
        lines = f.read().splitlines()

    last_epoch = ""
    for key, group in itertools.groupby(lines, key=lambda x: x != ''):
        if key:
            group_lines = list(group)
            if len(group_lines) == 4:
                last_epoch = group_lines

    m = re.search(fr"(?s:.*){metric}: (\d?\d\.\d\d)%", last_epoch[2])
    if m:
        return float(m.group(1)) / 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="Where to write the checkpoints and results during fine-tuning.")
    args = parser.parse_args()

    result_funcs = {
        'pos': get_transformer_results,
        'ner': get_transformer_results,
        'dp': get_diaparser_results,
        'ats': get_transformersum_results
    }
    metrics = {'pos': 'predict_accuracy', 'ner': 'predict_f1', 'dp': 'LAS', 'ats': 'r2-recall'}
    root_dir = args.root_dir

    results = {}
    for task in result_funcs:
        task_dir = os.path.join(root_dir, task)
        if os.path.exists(task_dir):
            runs = [name for name in os.listdir(task_dir) if name.isdigit()]

            task_results = {}
            for run in runs:
                run_dir = os.path.join(task_dir, run)
                task_results[run] = result_funcs[task](run_dir, metrics[task])

            results[task] = task_results

    print("All scores:")
    for task, task_results in results.items():
        print(f"{task:>15}", " ".join([f"{score:>7.2%}" for run, score in task_results.items()]))

    print("\nAverages:")
    for task, task_results in results.items():
        avg = sum(task_results.values()) / len(task_results.values())
        print(f"{task:>15} {avg:>7.2%}")


if __name__ == '__main__':
    main()
