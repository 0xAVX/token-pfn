# TokenPFN — subword vision for TabPFN without an LLM

Real tables contain text (reviews, merchants, SKUs). Either pay for an LLM
embedding or accept TF-IDF mush. TokenPFN takes a third path: a stock
subword tokenizer (no transformer inference) → **token sketch** (hashed
n-gram geometry) + **token evidence** (cross-fit per-token log-odds
aggregates) → TabPFN-3.5 reasons jointly over structured + lexical features.

TabPFN answers *how lexical evidence interacts with structured context*;
the tokenizer answers *what lexical evidence is in the cell*.

## Results (Women's Clothing Reviews, Recommended IND, `figs/tokenpfn.csv`)

| setup | AUC |
|---|---|
| A structured-only | 0.9721 |
| B native TRANSFORM_TEXT (char n-gram→SVD) | 0.9793 |
| C sketch | 0.9774 |
| D evidence | **0.9816** |
| E sketch+evidence | **0.9817** |

Evidence beats the built-in text path; sketch alone doesn't. Note on
Tokenizers v1: v1 preserves token IDs by design, so it changes speed, not
model inputs — our contribution is the modeling (venv ships tokenizers 0.23;
v1 is a drop-in speedup when stable).

## Reproduce

```bash
# data: kaggle datasets download -d nicapotato/womens-ecommerce-clothing-reviews -p data --unzip
<venv-python> -m pytest tests/ -q
<venv-python> experiments/run.py   # figs/tokenpfn.csv
<venv-python> demo/app.py          # lexicon view (port 5005)
```
