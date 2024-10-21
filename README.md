# IceEval

![Version](https://img.shields.io/badge/Version-main-green)
![Python](https://img.shields.io/badge/python-3.9-blue?logo=python&logoColor=white)
![CI Status](https://img.shields.io/badge/CI-[unavailable]-red)
![Docker](https://img.shields.io/badge/Docker-[unavailable]-red)

IceEval is a benchmark for evaluating and comparing the quality of pre-trained language models (LM's).

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

## Leaderboard

On the **[Leaderboard](doc/results.md)** page, you can find current results of the most important Icelandic language models (LMs).


## Description

IceEval is foremost a benchmark for evaluating and comparing the quality of pre-trained language models. The models are evaluated on a selection of four NLP tasks for Icelandic: part-of-speech (PoS) tagging, named entity recognition (NER), dependency parsing (DP) and automatic text summarization (ATS).

But you can also use IceEval as a demonstration, how to fine-tune any of the evaluated LM's on specific token-classification tasks. We used the HuggingFace Transformers [Token-Classification example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) as our baseline.

IceEval includes scripts for downloading the datasets, splitting them into training, validation and test splits and training and evaluating models for each task. The benchmark uses the Transformers, DiaParser and TransformerSum libraries for fine-tuning and evaluation.

For PoS tagging, the models are evaluated on the MIM-GOLD corpus, and the benchmark reports tagging accuracy.<br>
For NER, the models are evaluated on the MIM-GOLD-NER corpus, and the benchmark reports F1 scores.<br>
For DP, the models are evaluated on the IcePaHC-UD corpus, and the benchmark reports labeled attachment scores (LAS).<br>
For ATS, the models are evaluated on the IceSum corpus, and the benchmark reports ROUGE-2 recall scores.

MIM-GOLD 21.05 - https://repository.clarin.is/repository/xmlui/handle/20.500.12537/114<br>
MIM-GOLD-NER 2.0 - https://repository.clarin.is/repository/xmlui/handle/20.500.12537/230<br>
IcePaHC-UD - https://github.com/UniversalDependencies/UD_Icelandic-IcePaHC/<br>
IceSum 22.09 - https://repository.clarin.is/repository/xmlui/handle/20.500.12537/285<br>

Transformers - https://github.com/huggingface/transformers<br>
DiaParser - https://github.com/Unipisa/diaparser<br>
TransformerSum - https://github.com/HHousen/TransformerSum<br>

**Note:**<br>
The TransformerSum library uses a `ROUGE` package that isn't Unicode-friendly. This mimics the original `ROUGE` package for Perl which wasn't Unicode friendly either. When calculating `ROUGE` scores, first the GOLD and predicted summaries are pre-processed by discarding all non-alphanumeric characters using the regular expression `[^a-z0-9]+`. This would result in all accented characters being replaced by spaces, which leads to much lower `ROUGE` scores. In our bundled version of `TransformerSum`, this issue has been fixed. English-oriented stemming on the summaries is also fixed.

## Installation

Install all required packages:

``` shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage

Download and process the datasets:

``` shell
python3 download_datasets.py
```

Run the benchmark:

``` shell
python3 finetune.py --model_path <path_to_model> --model_type <model_type> --output_dir <output_dir>
```

The argument `--path_to_model` specifies the path to pretrained model or model identifier from huggingface.co/models. You can also use the syntax `<group>/<model>@<revision>` to specify a certain version on HuggingFace.<br>
`--model_type` should identify the type of the model (e.g., 'bert', 'roberta' or 'electra').<br>
`--output_dir` specifies the root directory where the checkpoints and results should be saved.

Print results:

``` shell
python3 get_results.py <root_dir>
```

The `root_dir` argument specifies an output folder that was generated when running the benchmark with finetune.py. You can also run this script while the fine-tuning process is ongoing.

Generate detailed results for PoS-Tagging:

```bash
python3 eval_pos.py --splits_dir <path-to-mim-gold> \
          --results_dir <root_dir/pos> \
          --output_dir <output_dir> \
          --ignore_tags <space separated list of tags to be excluded from eval, e.g. B-x B-e> \
          --confusion_matrix \
          --full
```

This will generate results of the model as graphical confusion matrices of the top 30 mis-classifications and additional text files with more details inside a subfolder `confusions/` inside the given output directory.<br>
If option `--full` is given, the full set of test data is used for evaluation, instead of just the part of the test data that fits into the max. input sequence of the model.<br>
Optionally, parameter `--ignore_tags` can be used to specify a list of tags that should not be used for the evaluation. Note however, that the model still can predict these tags, and therefore you either need to train the model excluding these tags or need to modify the weights of these tags inside the classifier layer of the trained model to get the highest prediction accuracies. 

## License

Copyright 2022 Reykjavik University

Licensed under the Apache License, Version 2.0 (the "License"), see file [LICENSE](LICENSE) for exact terms;

- The included package [TransformerSum](https://github.com/HHousen/TransformerSum) is licensed under [GPLv3.0](https://github.com/HHousen/TransformerSum/blob/master/LICENSE).
- The included package [DiaParser](https://github.com/Unipisa/diaparser) is licensed under [MIT](https://github.com/Unipisa/diaparser/blob/master/LICENSE).

The effective license of this package, taking into account all dependent package licenses is [GPLv3.0](https://github.com/HHousen/TransformerSum/blob/master/LICENSE). If you want to base your work on this project, but cannot comply with the terms of GPLv3, you must remove any code depending on `TransformerSum`. In that case, the effective license is Apache-2.0.

## Acknowledgements

This project was funded by the Language Technology Programme for Icelandic 2019-2023. The programme, which is managed and coordinated by
[Almannarómur](https://almannaromur.is/), was funded by the Icelandic Ministry of Education, Science and Culture.
