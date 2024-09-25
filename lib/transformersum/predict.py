#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import re
import subprocess
from collections import defaultdict
# from tokenizer import split_into_sentences
from extractive import ExtractiveSummarizer


# def split(text):
#     lines = [line.strip() for line in re.split(r'[\n\r]+', text.strip())]
#     return [s.strip() for line in lines for s in split_into_sentences(line) if line]


def main():
    test_path = r"C:\Jon\Resources\Summarization\icesum\transformersum\data\electra-base\test.mbl.txt"
    test_set = []
    with open(test_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                test_set.append(entry)

    output = defaultdict(dict)
    log_dir = '../lightning_logs'

    for version_dir in os.listdir(log_dir):
        if version_dir != 'version_0':
            continue

        ckpt_dir = os.path.join(log_dir, version_dir, 'checkpoints')
        version = re.match(r'.*(version_\d+)', version_dir).group(1)

        for model_name in os.listdir(ckpt_dir):
            out_name = f"{version}_{model_name}"
            model_path = os.path.join(ckpt_dir, model_name)
            model = ExtractiveSummarizer.load_from_checkpoint(model_path, dataloader_num_workers=0, max_seq_length=512)

            for entry in test_set:
                text = entry['source']
                sent_lengths = entry['sent_lengths']
                print(text)
                index = {s: pos for pos, s in enumerate(text)}
                prediction = model.predict_sentences(text, True, 10, True, sent_lengths)

                print(prediction)
                x = input()

                summary = []
                length = 0

                for s, score in sorted(prediction, key=lambda x: x[1], reverse=True):
                    length += len(s.split())
                    summary.append(s)
                    if length > 100:
                        break
                # summary.sort(key=index.get)
                # output[out_name][eid] = summary

    # output_path = os.path.join(r'C:\Jon\Projects\summarization\predictions\tfsum-136.json')
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
