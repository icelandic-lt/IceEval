## Leaderboard of Icelandic LM's

This overview shows our evaluation results for the most important Icelandic Language Models available via HuggingFace.

Each model evaluation has been run in `October/2024` on a single GPU card with **script default values**.<br>

**These scores are evaluated after a constant training length for each model and represent the lower bounds. Higher values are possible, if training time is higher/unbound and early stopping is applied.**<br>

- The score values show for PoS: Accuracy, NER: f1, DP: LAS, ATS: ROUGE-2 recall
- The given **rank is calculated by averaging all final scores** and is an orientation for the best performing all-rounders
- Highest values of individual categories are written in **bold** letters, lowest in *italic* letters. Additionally, ↑↓ mark the highest/lowest value in each category

| Rank | Model                                                                                           | PoS`*`          | NER          | DP           | ATS`**`      | Parameters       | Model-Size       | License                                                           |
|------|------------------------------------------------------------------------------------------------|-----------------|--------------|--------------|--------------|------------------|------------------|-------------------------------------------------------------------|
| 1    | [MaCoCu/XLMR-MaCoCu-is](https://huggingface.co/MaCoCu/XLMR-MaCoCu-is)                           | **98.00%** ↑`¹` | 92.11% `²`   | **84.69%** ↑ | **71.58%** ↑ | **559.89 Mio** ↑ | **2135.82 MB** ↑ | [CC0 1.0 Universal](https://choosealicense.com/licenses/cc0-1.0/) |
| 2    | [icelandic-lt/electra-base-igc-is](https://huggingface.co/icelandic-lt/electra-base-igc-is)     | 97.67%          | 90.96%       | 84.10%       | 71.24%       | 110.11 Mio       | 420.03 MB        | [CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/)       |
| 3    | [icelandic-lt/convbert-base-igc-is](https://huggingface.co/icelandic-lt/convbert-base-igc-is)   | 97.71%          | 91.52%       | 84.34%       | 69.09%       | 106.90 Mio       | 407.78 MB        | [CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/)       |
| 4    | [MaCoCu/XLMR-base-MaCoCu-is](https://huggingface.co/MaCoCu/XLMR-base-MaCoCu-is)                 | 97.23%          | 89.45%       | 83.12%       | 70.07%       | 278.04 Mio       | 1060.66 MB       | [CC0 1.0 Universal](https://choosealicense.com/licenses/cc0-1.0/) |
| 5    | [mideind/IceBERT](https://huggingface.co/mideind/IceBERT)                                       | 97.88%          | **92.40%** ↑ | 78.87%       | 69.91%       | 124.44 Mio       | 474.72 MB        | [CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/)           |
| 6    | [icelandic-lt/convbert-small-igc-is](https://huggingface.co/icelandic-lt/convbert-small-igc-is) | 96.94%          | 89.95%       | 82.87%       | 68.74%       | 21.52 Mio        | 82.12 MB         | [CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/)       |
| 7    | [mideind/IceBERT-large](https://huggingface.co/mideind/IceBERT-large)                           | 97.69%          | 90.98%`³`    | 78.41%       | 71.20%       | 355.09 Mio       | 1354.56 MB       | [CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/)           |
| 8    | [icelandic-lt/electra-small-igc-is](https://huggingface.co/icelandic-lt/electra-small-igc-is)   | 96.86%          | *88.00%* ↓   | 82.10%       | *68.67%* ↓   | *13.69 Mio* ↓    | *52.21 MB* ↓     | [CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/)       |
| 9    | [mideind/IceBERT-igc](https://huggingface.co/mideind/IceBERT-igc)                               | 97.36%          | 90.27%       | *78.10%* ↓   | 69.67%       | 124.44 Mio       | 474.72 MB        | [CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/)           |
| 10   | [jonfd/electra-small-is-no](https://huggingface.co/jonfd/electra-small-is-no)                   | *96.63%* ↓      | 88.19%       | 81.66%       | 68.86%       | 17.78 Mio        | 67.84 MB         | [CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/)       |

`*` For possible upper bounds of this value, see paper [Is Part-of-Speech Tagging a Solved Problem for Icelandic?](https://aclanthology.org/2023.nodalida-1.8.pdf)<br>
`**` Best value from 2 runs with either 5 or 10 epochs training<br>
`¹` Model achieves SOTA results (98.19% accuracy when excluding evaluation for x, e tags) for **single-label PoS tagging** in Icelandic just with standard fine-tuning procedures<br>
`²` Fold 5 training was not converging with the standard seed 42, therefore the seed parameter was changed to 43 in that particular case<br>
`³` Same as for `²` but for folds 5, 6, 7

Besides absolute scores for a specific task, please take also into account the license of the specific model and its size / number of model parameters.<br>
The inference speed is roughly anti-proportional to the model's parameter count.

**If you think, that an important model is missing from this evaluation or if you like to inquire for specific evaluation results or fine-tuned models, please open an [issue](https://github.com/icelandic-lt/IceEval/issues) inside this repository.**
