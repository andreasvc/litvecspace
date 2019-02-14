# Vector space explorations of literary language
Accompanying code for the paper "[Vector space explorations of literary language](https://doi.org/10.1007/s10579-018-09442-4)"

This repository is intended for documentation purposes, as the relevant data cannot be made publicly available.

## Data used

- The Riddle of Literary Quality corpus of 401 novels and survey data; http://literaryquality.huygens.knaw.nl/
- Topic model presented in http://dh2016.adho.org/abstracts/95

## Requirements

See `requirements.txt`.
Install with `pip3 install -r requirements.txt`

## Overview

- features.py: divides corpus into chunks and computes BoW features and paragraph vectors
- notebook.ipynb: fit predictive models, data analysis, tables, figures, etc.

## Reference

```latex
@article{vancranenburgh2019vecspace,
    author={van Cranenburgh, Andreas
            and van Dalen-Oskam, Karina
            and van Zundert, Joris},
    title={Vector space explorations of literary language},
    year={2019},
    journal={Language Resources and Evaluation},
    month={Feb},
    day={09},
    issn={1574-0218},
    doi={10.1007/s10579-018-09442-4},
    url={https://doi.org/10.1007/s10579-018-09442-4}
}
```
