#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime
import os
import re
import shutil
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PYTHON = sys.executable

def get_cuda_device():
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_devices is not None:
        devices = cuda_devices.split(',')
        if devices:
            return devices[0]  # Return the first available device
    return '0'  # Default to '0' if CUDA_VISIBLE_DEVICES is not set or empty


def split_model_and_revision(model_path):
    match = re.match(r'(.*?)(?:@(.*))?$', model_path)
    if match:
        return match.group(1), match.group(2)
    return model_path, None

def finetune_pos(model_path, train_path, test_path, output_path, task, **kwargs):
    """Finetune a pre-trained language model on a token classification task.

    :param model_path: Path to pretrained model or model identifier from huggingface.co/models.
    :param train_path: The input training data file (a csv or JSON file).
    :param test_path: The input test data file to predict on (a csv or JSON file).
    :param output_path: The output directory where the model predictions and checkpoints will be written.
    :param task: The name of the task.
    :param kwargs: Additional keyword arguments.
    :return:
    """
    if os.path.exists(output_path):
        if kwargs.get('force', False):
            shutil.rmtree(output_path)
        else:
            logging.info(f"Already exists: {output_path}")
            return

    model_name, revision = split_model_and_revision(model_path)
    cmd = [PYTHON,
           '-X', 'utf8',
           'run_ner.py',
           '--task_name', task,
           '--model_name_or_path', model_name,
           '--tokenizer_name', model_name,
           '--output_dir', output_path,
           '--train_file', train_path,
           '--test_file', test_path,
           '--do_train',
           '--do_predict',
           '--text_column_name', 'tokens',
           '--label_column_name', 'tags',
           '--learning_rate', str(kwargs.get('learning_rate')),
           '--per_device_train_batch_size', str(kwargs.get('batch_size')),
           '--per_device_eval_batch_size', '256',
           '--eval_accumulation_steps', '8',
           '--save_steps', '0',
           '--num_train_epochs', str(kwargs.get('epochs')),
           '--seed', str(kwargs.get('seed')),
           '--logging_step', '500',
           '--overwrite_cache',
           ]

    logging.info(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Error in finetune_pos for {output_path}:")
        logging.error(result.stderr)
        raise RuntimeError(f"Subprocess failed: finetune_pos for {output_path}")


def finetune_dp(model_path, train_path, val_path, test_path, output_path, **kwargs):
    """Finetune a pre-trained language model on dependency parsing.

    :param model_path: Path to pretrained model or model identifier from huggingface.co/models.
    :param train_path: The input training data file (a csv or JSON file).
    :param val_path: The input evaluation data file to evaluate on (a csv or JSON file).
    :param test_path: The input test data file to predict on (a csv or JSON file).
    :param output_path: The output directory where the checkpoints and log will be written.
    :param kwargs: Additional keyword arguments.
    :return:
    """
    if os.path.exists(output_path):
        if kwargs.get('force', False):
            shutil.rmtree(output_path, ignore_errors=True)
        else:
            logging.info(f"Already exists: {output_path}")
            return

    model_name, revision = split_model_and_revision(model_path)
    device = get_cuda_device()
    cmd = [PYTHON,
           '-X', 'utf8',
           '-m', 'diaparser.cmds.biaffine_dependency', 'train',
           '--epochs', str(kwargs.get('epochs')),
           '--build',
           '--device', device,
           '--path', output_path,
           '--seed', str(kwargs.get('seed')),
           '--train', train_path,
           '--dev', val_path,
           '--test', test_path,
           '--feat', 'bert',
           '--bert', model_name,
           ]

    logging.info(" ".join(cmd))
    result = subprocess.run(cmd, cwd=kwargs.get('cwd'), capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Error in finetune_dp for {output_path}:")
        logging.error(result.stderr)
        raise RuntimeError(f"Subprocess failed: finetune_dp for {output_path}")


def finetune_ats(model_path, model_type, data_path, output_path, **kwargs):
    """Finetune a pre-trained language model on extractive text summarization.

    :param model_path: Path to pretrained model or model identifier from huggingface.co/models.
    :param model_type: The type of the model (e.g., 'bert', 'roberta' or 'electra').
    :param data_path: Path to the training, validation and test files.
    :param output_path: The output directory where the checkpoints and results will be written.
    :param kwargs: Additional keyword arguments.
    """
    if os.path.exists(output_path):
        if kwargs.get('force', False):
            shutil.rmtree(output_path)
        else:
            logging.info(f"Already exists: {output_path}")
            return

    model_name, revision = split_model_and_revision(model_path)
    cmd = [PYTHON,
           '-X', 'utf8',
           'lib/transformersum/main.py',
           '--model_name_or_path', model_name,
           '--model_type', model_type,
           '--data_path', data_path,
           '--data_type', 'txt',
           '--default_root_dir', output_path,
           '--do_train',
           '--do_test',
           '--max_epochs', str(kwargs.get('epochs')),
           '--use_logger', 'tensorboard',
           '--batch_size', str(kwargs.get('batch_size')),
           '--classifier', kwargs.get('classifier'),
           '--no_use_token_type_ids',
           '--no_test_block_trigrams',
           '--pooling_mode', kwargs.get('pooling_mode'),
           '--seed', str(kwargs.get('seed'))]

    logging.info(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Error in finetune_ats for {output_path}:")
        logging.error(result.stderr)
        raise RuntimeError(f"Subprocess failed: finetune_ats for {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path",
                        help="Path to pretrained model or model identifier from huggingface.co/models.", required=True)
    parser.add_argument("-t", "--model_type",
                        help="The type of the model (e.g., bert, roberta or electra).", required=True)
    parser.add_argument("-o", "--output_dir", help="Where to write the checkpoints and results during fine-tuning.",
                        required=True)
    parser.add_argument("-f", "--force", action="store_true", default=False, help="Force overwrite of existing output directories")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tasks", nargs='+', default=['pos', 'ner', 'dp', 'ats'],
                        help="Tasks to run. Can be any combination of 'pos', 'ner', 'dp', 'ats'")
    args = parser.parse_args()

    task_args = {
        'pos': {'epochs': 20, 'batch_size': 16, 'learning_rate': 5e-5, 'task': 'pos'},
        'ner': {'epochs': 10, 'batch_size': 16, 'learning_rate': 5e-5, 'task': 'ner'},
        'dp': {'epochs': 15},
        'ats': {'epochs': 5, 'batch_size': 8, 'classifier': 'linear', 'pooling_mode': 'mean_tokens'}
        # note: for ats, some models profit for setting epochs to a higher value, e.g. 10
    }

    model_path = args.model_path
    model_type = args.model_type
    force = args.force
    seed = args.seed
    tasks = args.tasks

    # Validate tasks
    valid_tasks = {'pos', 'ner', 'dp', 'ats'}
    invalid_tasks = set(tasks) - valid_tasks
    if invalid_tasks:
        logging.error(f"Invalid tasks specified: {', '.join(invalid_tasks)}")
        sys.exit(1)

    # Default dataset path is the 'dataset' directory where this script is located
    script_path = os.path.realpath(__file__)
    root_dir = os.path.dirname(script_path)
    lib_dir = os.path.join(root_dir, 'lib')

    dataset_paths = {
        'pos': os.path.join(root_dir, 'datasets', 'mim-gold'),
        'ner': os.path.join(root_dir, 'datasets', 'mim-gold-ner'),
        'dp': os.path.join(root_dir, 'datasets', 'icepahc-ud'),
        'ats': os.path.join(root_dir, 'datasets', 'icesum')
    }

    # Output dir
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        # If no output directory was specified, default to the current date and time
        name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        output_dir = os.path.join(root_dir, 'runs', name)

    # Make sure the output directory is absolute
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)

    folds = [f'{n:0>2}' for n in range(1, 11)]
    runs = [f'{n:0>2}' for n in range(1, 6)]

    try:
        # Part-of-speech tagging
        if 'pos' in tasks:
            dataset_dir = dataset_paths['pos']

            for fold in folds:
                train_path = os.path.join(dataset_dir, f'{fold}-train.json')
                test_path = os.path.join(dataset_dir, f'{fold}-test.json')
                output_path = os.path.join(output_dir, 'pos', fold)
                finetune_args = task_args['pos'].copy()
                finetune_args.update({'force': force, 'seed': seed})

                finetune_pos(model_path, train_path, test_path, output_path, **finetune_args)

        # Named entity recognition
        if 'ner' in tasks:
            dataset_dir = dataset_paths['ner']

            for fold in folds:
                train_path = os.path.join(dataset_dir, f'{fold}-train.json')
                test_path = os.path.join(dataset_dir, f'{fold}-test.json')
                output_path = os.path.join(output_dir, 'ner', fold)
                finetune_args = task_args['ner'].copy()
                finetune_args.update({'force': force, 'seed': seed})

                finetune_pos(model_path, train_path, test_path, output_path, **finetune_args)

        # Dependency parsing
        if 'dp' in tasks:
            dataset_dir = dataset_paths['dp']

            train_path = os.path.join(dataset_dir, 'is_icepahc-ud-train.conllu')
            val_path = os.path.join(dataset_dir, 'is_icepahc-ud-dev.conllu')
            test_path = os.path.join(dataset_dir, 'is_icepahc-ud-test.conllu')
            finetune_args = task_args['dp'].copy()
            finetune_args.update({'force': force, 'seed': seed, 'cwd': lib_dir})

            for num_run in runs:
                output_path = os.path.join(output_dir, 'dp', num_run, num_run)

                finetune_dp(model_path, train_path, val_path, test_path, output_path, **finetune_args)

        # Automatic text summarization
        if 'ats' in tasks:
            ats_original_dataset_dir = os.path.join(dataset_paths['ats'], 'transformersum')
            ats_dataset_dir = os.path.join(output_dir, 'ats', 'icesum')
            os.makedirs(ats_dataset_dir, exist_ok=True)

            for icesum_file in os.listdir(ats_original_dataset_dir):
                if icesum_file.endswith('.json'):
                    path_from = os.path.join(ats_original_dataset_dir, icesum_file)
                    path_to = os.path.join(ats_dataset_dir, icesum_file)
                    shutil.copy(path_from, path_to)

            finetune_args = task_args['ats'].copy()
            finetune_args.update({'force': force, 'seed': seed})

            for num_run in runs:
                output_path = os.path.join(output_dir, 'ats', num_run)
                finetune_ats(model_path, model_type, ats_dataset_dir, output_path, **finetune_args)
    except RuntimeError as e:
        logging.error(f"Script terminated due to an error: {str(e)}")
        sys.exit(1)

    logging.info("All tasks completed successfully.")

if __name__ == '__main__':
    main()
