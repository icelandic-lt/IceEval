# IceEval

![Version](https://img.shields.io/badge/Version-22.09-darkviolet)
![Python](https://img.shields.io/badge/python-3.9-blue?logo=python&logoColor=white)
![CI Status](https://img.shields.io/badge/CI-[unavailable]-red)
![Docker](https://img.shields.io/badge/Docker-[unavailable]-red)

IceEval is a benchmark for evaluating and comparing the quality of pre-trained language models.

## Overview
- **Language:** Python
- **Language Version/Dialect:**
  - Python: 3.9+
- **Category:** [Support Tools](https://github.com/icelandic-lt/icelandic-lt/blob/main/doc/st.md)
- **Domain:** Generic
- **Status:** Stable
- **Origins:** [IceEval - Icelandic Natural Language Processing Benchmark 22.09](http://hdl.handle.net/20.500.12537/297)

## System Requirements
- Operating System: Linux, e.g. Ubuntu
- NVIDIA GPUs with CUDA support

## Description

IceEval is a benchmark for evaluating and comparing the quality of pre-trained
language models. The models are evaluated on a selection of four NLP tasks for
Icelandic: part-of-speech (PoS) tagging, named entity recognition (NER),
dependency parsing (DP) and automatic text summarization (ATS). IceEval includes
scripts for downloading the datasets, splitting them into training, validation
and test splits and training and evaluating models for each task. The benchmark
uses the Transformers, DiaParser and TransformerSum libraries for fine-tuning
and evaluation.

For PoS tagging, the models are evaluated on the MIM-GOLD corpus, and the
benchmark reports tagging accuracy. For NER, the models are evaluated on the
MIM-GOLD-NER corpus, and the benchmark reports F1 scores. For DP, the models are
evaluated on the IcePaHC-UD corpus, and the benchmark reports labeled attachment
scores (LAS). For ATS, the models are evaluated on the IceSum corpus, and the
benchmark reports ROUGE-2 recall scores.

MIM-GOLD 21.05 - https://repository.clarin.is/repository/xmlui/handle/20.500.12537/114
MIM-GOLD-NER 2.0 - https://repository.clarin.is/repository/xmlui/handle/20.500.12537/230
IcePaHC-UD - https://github.com/UniversalDependencies/UD_Icelandic-IcePaHC/
IceSum 22.09 - https://repository.clarin.is/repository/xmlui/handle/20.500.12537/285

Transformers - https://github.com/huggingface/transformers
DiaParser - https://github.com/Unipisa/diaparser
TransformerSum - https://github.com/HHousen/TransformerSum

## Installation

Install all required packages:

``` shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage

Download and process the datasets:

``` shell
python download_datasets.py
```

Run the benchmark:

``` shell
python finetune.py --model_path <path_to_model> --model_type <model_type> --output_dir <output_dir>
```

The `path_to_model` argument specifies the path to pretrained model or model
identifier from huggingface.co/models. The model_type argument should identify
the type of the model (e.g., 'bert', 'roberta' or 'electra'). The `output_dir`
specifies the root directory where the checkpoints and results should be saved.

Print results:

``` shell
python get_results.py <root_dir>
```

The `root_dir` argument specifies an output folder that was generated when
running the benchmark with finetune.py.

## License

Copyright 2022 Reykjavik University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Acknowledgements

This project was funded by the Language Technology Programme for Icelandic
2019-2023. The programme, which is managed and coordinated by
[Almannarómur](https://almannaromur.is/), was funded by the Icelandic Ministry
of Education, Science and Culture.
