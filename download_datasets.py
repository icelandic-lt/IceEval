#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import itertools
import json
import logging
import os
import requests
import subprocess
import sys
import time
from zipfile import ZipFile


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PYTHON = sys.executable


def download(url, path, overwrite=False):
    if os.path.exists(path):
        logging.info(f"{path} already exists, not downloading. Use --force-download to redownload.")
    else:
        dest_dir, dest_file = os.path.split(path)
        os.makedirs(dest_dir, exist_ok=True)

        content = requests.get(url, stream=True).content
        logging.info(f"Downloading {url} to {path}...")
        with open(path, 'wb') as f:
            f.write(content)

        time.sleep(3)


def parse_tsv(document, max_fields, token_field=0, label_field=1):
    """Parse a document in a tab-seperated value (tsv) format.

    :param document: The .tsv document as a string.
    :param max_fields: The number of tab-seperated values each non-empty line should contain.
    :param token_field: The index of the field which contains the token.
    :param label_field: The index of the field which contains the label.
    :return: A list of sentences, each of which is represented with a dictionary of tokens and tags.
    """
    lines = document.splitlines()
    sentences = []

    for key, group in itertools.groupby(lines, key=lambda x: x != ''):
        if key:
            sentence = [line.strip().split('\t')[:max_fields] for line in group]
            s_json = {
                'tokens': [t[token_field] for t in sentence],
                'tags': [t[label_field] for t in sentence]
            }

            sentences.append(s_json)

    return sentences


def split_mim_gold_ner(mim_gold_ner):
    """Generate training/test splits for the MIM-GOLD-NER corpus which are identical to the MIM-GOLD splits.

    :param mim_gold_ner: The MIM-GOLD-NER corpus.
    :return: A dictionary of MIM-GOLD-TRAIN training/test files and list of the sentences they contain.
    """
    with open('mim-gold-ner-splits.json', encoding='utf-8') as f:
        splits = json.load(f)

    output_files = {}
    for split_file, indices in splits.items():
        output_sentences = []
        for file, start, end in indices:
            output_sentences.extend(mim_gold_ner[file][start:end])

        output_files[split_file] = output_sentences

    return output_files


def tsv_to_jsonl(files, output_dir):
    """Write documents read from .tsv files in a .jsonl format.

    :param files: A dictionary of filenames and list of the sentences they contain.
    :param output_dir: The directory where the converted files should be written.
    """
    for split_file, sentences in files.items():
        split_type = split_file[2:4]
        out_file = split_file[:2] + ('-train.json' if split_type == 'TM' else '-test.json')
        mgn_output_path = os.path.join(output_dir, out_file)

        with open(mgn_output_path, 'w', encoding='utf-8') as f:
            logging.debug(f"Writing file: {mgn_output_path}")
            for s in sentences:
                f.write(json.dumps(s, ensure_ascii=False, separators=(',', ':')) + '\n')


def compare_corpora(a, b):
    """Compare two corpora that should consist of the exact same sentences and log the results."""
    all_identical = True

    for file, a_sentences in a.items():
        b_sentences = b[file]

        file_identical = True

        if len(a_sentences) != len(b_sentences):
            file_identical = False
            all_identical = False
        else:
            for a_sent, b_sent in zip(a_sentences, b_sentences):
                if a_sent['tokens'] != b_sent['tokens']:
                    file_identical = False
                    all_identical = False
                    break

        if not file_identical:
            logging.warning(f"MIM-GOLD-NER: Contents of {file} differ from the MIM-GOLD version")

    if all_identical:
        logging.info("MIM-GOLD-NER: Successfully generated training and test splits")


def main():
    # Datasets:
    #   - MIM-GOLD 21.05 - https://repository.clarin.is/repository/xmlui/handle/20.500.12537/114
    #   - MIM-GOLD-NER 2.0 - https://repository.clarin.is/repository/xmlui/handle/20.500.12537/230
    #   - IcePaHC-UD - https://github.com/UniversalDependencies/UD_Icelandic-IcePaHC/
    #   - IceSum 22.09 - https://repository.clarin.is/repository/xmlui/handle/20.500.12537/285

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--force-download", action='store_true',
                        help="Whether to overwrite previously downloaded files.")
    args = parser.parse_args()

    dataset_urls = {
        'mim-gold': {
            'MIM-GOLD-SETS-21.05.zip': 'https://repository.clarin.is/repository/xmlui/bitstream/handle/'
                                       '20.500.12537/114/MIM-GOLD-SETS-21.05.zip?sequence=1&isAllowed=y'
        },
        'mim-gold-ner': {
            'MIM-GOLD-2_0.zip': 'https://repository.clarin.is/repository/xmlui/bitstream/handle/'
                                '20.500.12537/230/MIM-GOLD-2_0.zip?sequence=15&isAllowed=y'
        },
        'icepahc-ud': {
            'is_icepahc-ud-dev.conllu': 'https://raw.githubusercontent.com/UniversalDependencies/UD_Icelandic-IcePaHC/r2.10/is_icepahc-ud-dev.conllu',
            'is_icepahc-ud-test.conllu': 'https://raw.githubusercontent.com/UniversalDependencies/UD_Icelandic-IcePaHC/r2.10/is_icepahc-ud-test.conllu',
            'is_icepahc-ud-train.conllu': 'https://raw.githubusercontent.com/UniversalDependencies/UD_Icelandic-IcePaHC/r2.10/is_icepahc-ud-train.conllu'
        },
        'icesum': {
            'icesum.zip': 'https://repository.clarin.is/repository/xmlui/bitstream/handle/'
                          '20.500.12537/285/icesum.zip?sequence=3&isAllowed=y'
        }
    }

    # Download datasets
    for dataset, data_urls in dataset_urls.items():
        dest_dir = os.path.join('datasets', dataset)
        for filename, url in data_urls.items():
            dest_path = os.path.join(dest_dir, filename)
            download(url, dest_path, overwrite=args.force_download)

    # MIM-GOLD - Process
    pos_dataset_dir = os.path.join('datasets', 'mim-gold')
    pos_archive_path = os.path.join(pos_dataset_dir, 'MIM-GOLD-SETS-21.05.zip')
    pos_original = {}
    with ZipFile(pos_archive_path) as z:
        tsv_paths = [name for name in z.namelist() if name.endswith('tsv')]
        for path in tsv_paths:
            tsv_dir, tsv_filename = os.path.split(path)
            logging.debug(f"MIM-GOLD: Reading {path}")
            with z.open(path) as f:
                doc = f.read().decode('utf-8')

            parsed = parse_tsv(doc, max_fields=3, token_field=0, label_field=2)

            # Seqeval will complain unless POS tags start with a 'B-'
            for sentence in parsed:
                sentence['tags'] = ['B-' + t for t in sentence['tags']]

            pos_original[tsv_filename] = parsed

    tsv_to_jsonl(pos_original, pos_dataset_dir)

    # MIM-GOLD-NER - Process
    ner_dataset_dir = os.path.join('datasets', 'mim-gold-ner')
    ner_archive_path = os.path.join(ner_dataset_dir, 'MIM-GOLD-2_0.zip')
    ner_original = {}
    with ZipFile(ner_archive_path) as z:
        txt_paths = [name for name in z.namelist() if name.endswith('txt')]
        for path in txt_paths:
            tsv_dir, tsv_filename = os.path.split(path)
            logging.debug(f"MIM-GOLD-NER: Reading {path}")
            with z.open(path) as f:
                doc = f.read().decode('utf-8')

            ner_original[tsv_filename] = parse_tsv(doc, max_fields=2, token_field=0, label_field=1)

    ner_splits = split_mim_gold_ner(ner_original)
    compare_corpora(pos_original, ner_splits)
    tsv_to_jsonl(ner_splits, ner_dataset_dir)

    # IceSum - Process
    icesum_dataset_dir = os.path.join('datasets', 'icesum')
    icesum_archive_path = os.path.join(icesum_dataset_dir, 'icesum.zip')
    icesum_files = ["icesum/icesum.json", "icesum/process_transformersum.py", "icesum/splits.json"]

    with ZipFile(icesum_archive_path) as z:
        for icesum_file in icesum_files:
            filename = os.path.split(icesum_file)[1]
            output_path = os.path.join(icesum_dataset_dir, filename)
            with z.open(icesum_file) as f_in:
                contents = f_in.read()

                with open(output_path, 'wb') as f_out:
                    f_out.write(contents)

    cmd = [PYTHON, 'process_transformersum.py']
    logging.info(f"IceSum: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=icesum_dataset_dir)


if __name__ == '__main__':
    main()
