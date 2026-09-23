# TokenPFN — subword vision for TabPFN without an LLM

Real tables contain text (reviews, merchants, SKUs). Either pay for an LLM
embedding or accept TF-IDF mush. TokenPFN takes a third path: a stock
subword tokenizer (no transformer inference) → **token sketch** (hashed
n-gram geometry) + **token evidence** (cross-fit per-token log-odds
aggregates) → TabPFN-3.5 reasons jointly over structured + lexical features.

TabPFN answers *how lexical evidence interacts with structured context*;
the tokenizer answers *what lexical evidence is in the cell*.

## Results (Women's Clothing Reviews, Recommended IND)

Token evidence uses class-prior-normalized log-odds (not raw counts, so the
82%-positive base rate can't leak into scores). Fairness: evidence is built
from exactly the 12k context labels TabPFN trains on — no extra labels. 3 seeds:

| setup | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| A structured-only | 0.9721 | 0.9741 | 0.9734 | 0.9732 |
| B native TRANSFORM_TEXT | 0.9793 | 0.9835 | 0.9828 | 0.9819 |
| C sketch | 0.9774 | 0.9805 | 0.9806 | 0.9795 |
| D evidence | 0.9814 | 0.9849 | 0.9841 | **0.9835** |
| E sketch+evidence | 0.9814 | 0.9847 | 0.9847 | **0.9836** |

Evidence beats the built-in text path on every seed (+0.0016 mean); sketch
alone doesn't. Small but consistent — the demo leans on inspectability, not
just the gap. Note on Tokenizers v1: v1 preserves token IDs by design, so it
changes speed, not model inputs — our contribution is the modeling (venv
ships tokenizers 0.23; v1 is a drop-in speedup when stable).

## Reproduce

Fresh-env verified 2026-09-22 (clean venv, `pip install -e .`, TokenPFN suite
1 passed in 2s; bert tokenizer from public HF; TabPFN weights public, no keys).

```bash
# data: kaggle datasets download -d nicapotato/womens-ecommerce-clothing-reviews -p data --unzip
<venv-python> -m pytest tests/ -q
<venv-python> experiments/run.py 0   # figs/tokenpfn_s0.csv (seeds 0/1/2)
<venv-python> experiments/run.py 1
<venv-python> experiments/run.py 2
<venv-python> demo/app.py          # lexicon view (port 5005)
```
