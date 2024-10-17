## Results for Icelandic LM's

This overview shows our evaluation results for various Icelandic LM's available via HuggingFace.

Each model evaluation has been run in `October/2024` on a single GPU card with script default values.<br>
**These scores are evaluated after a constant training length for each model and represent the lower bounds that are possible, if training would continue.**<br>

The given **rank is calculated by averaging all final scores** and is an orientation for the best performing all-rounders.<br>
Missing values are from models currently under evaluation.

| Model | License | Parameters | Model-Size | PoS`*` | NER | DP`**` | ATS | Rank |
|-------|---------|------------|------------|-----|-----|----|----|------|
| jonfd/convbert-base-igc-is | CC-BY-4.0 | 106.90 Mio | 407.78 MB | 97.71% | 91.52% | 79.84% | 69.09% | 2 |
| jonfd/convbert-small-igc-is | CC-BY-4.0 | 21.52 Mio | 82.12 MB | 96.94% | 89.95% | 79.46% | 68.74% | 3 |
| jonfd/electra-base-igc-is | CC-BY-4.0 | 110.11 Mio | 420.03 MB | 97.67% | 90.96% | 81.11% | 70.80% | 1 |
| jonfd/electra-small-igc-is | CC-BY-4.0 | 13.69 Mio | 52.21 MB | 96.86% | 88.00% | 78.40% | 68.67% | 5 |
| jonfd/electra-small-is-no | CC-BY-4.0 | 17.78 Mio | 67.84 MB | | | | | |
| jonfd/gpt2-igc-is | CC-BY-4.0 | 125.01 Mio | 488.88 MB | | | | | |
| MaCoCu/XLMR-base-MaCoCu-is | CC0 1.0 Universal | 278.04 Mio | 1060.66 MB | | | | | |
| MaCoCu/XLMR-MaCoCu-is | CC0 1.0 Universal | 559.89 Mio | 2135.82 MB | | | | | |
| mideind/IceBERT | AGPLv3 | 124.44 Mio | 474.72 MB | 97.88% | 92.40% | 72.51% | 69.91% | 4 |
| mideind/IceBERT-igc | AGPLv3 | 124.44 Mio | 474.72 MB | | | | | |
| mideind/IceBERT-large | AGPLv3 | 355.09 Mio | 1354.56 MB | | | | | |

`*` For a possible upper bound of this value, see the paper [Is Part-of-Speech Tagging a Solved Problem for Icelandic?](https://aclanthology.org/2023.nodalida-1.8.pdf)<br>
`**` Note that results for DiaParser are significantly lower than expected, need investigation!


Besides absolute scores for a specific use-case, please take also into account the license of the specific model and its size / number of model parameters.<br>
The runtime is roughly anti-proportional to the model's parameter count.

**If you think, that an important model is missing from this evaluation, open an [issue](https://github.com/icelandic-lt/IceEval/issues) inside this repository, and we try to add the missing info.**
