Description
-----------

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


Instructions
------------

1) Install all required packages:

pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

2) Download and process the datasets:

python download_datasets.py

3) Run the benchmark:

python finetune.py --model_path <path_to_model> --model_type <model_type> --output_dir <output_dir>

The path_to_model argument specifies the path to pretrained model or model
identifier from huggingface.co/models. The model_type argument should identify
the type of the model (e.g., 'bert', 'roberta' or 'electra'). The output_dir
specifies the root directory where the checkpoints and results should be saved.

4) Print results:

python get_results.py <root_dir>

The root_dir argument specifies an output folder that was generated when running
the benchmark with finetune.py.