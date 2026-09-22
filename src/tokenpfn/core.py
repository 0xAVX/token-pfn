"""TokenPFN core: subword tokenization -> sketch + cross-fit evidence features.

No transformer inference, no sentence embeddings. A stock tokenizer
(bert-base-uncased WordPiece) turns text cells into token streams; we build:
  sketch:    hashed unigram/bigram count features (unsupervised geometry).
  evidence:  per-token log-odds from OUT-OF-FOLD labels -> per-cell aggregates
             (pos/neg max, mean, top-3, variance, rarity, unk fraction, length,
             lexical diversity). Supervised signal, leak-free by construction.
TabPFN-3.5 does the joint structured+lexical reasoning downstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tokenizers import Tokenizer

_SKETCH_D = 64


def get_tokenizer():
    return Tokenizer.from_pretrained("bert-base-uncased")


def encode(texts: list[str], tok: Tokenizer):
    return tok.encode_batch([t if isinstance(t, str) else "" for t in texts])


def sketch_matrix(encs, d=_SKETCH_D):
    """Hashed unigram + bigram counts -> (n, 2d) non-negative ints."""
    n = len(encs)
    out = np.zeros((n, 2 * d), dtype=np.float32)
    for i, e in enumerate(encs):
        ids = e.ids
        for t in ids:
            out[i, t % d] += 1
        for a, b in zip(ids, ids[1:]):
            out[i, d + ((a * 31 + b) % d)] += 1
    return out


EV_COLS = ["ev_pos_max", "ev_neg_min", "ev_mean", "ev_top3",
           "ev_var", "ev_rarity", "ev_unk", "ev_len", "ev_lexdiv"]


def _cell_stats(ids, ev, logfreq, vocab_size, alpha=1.0):
    if not ids:
        return [0.0] * len(EV_COLS)
    v = np.array([ev.get(t, 0.0) for t in ids])
    pos = v[v > 0]
    top3 = np.sort(v)[-3:].mean() if len(v) else 0.0
    rare = np.mean([-logfreq.get(t, np.log(vocab_size)) for t in ids])
    unk = float(np.mean([t == 100 for t in ids]))  # [UNK] in bert vocab
    return [float(v.max()), float(v.min()), float(v.mean()), float(top3),
            float(v.var()), float(rare), unk, float(len(ids)),
            float(len(set(ids)) / len(ids))]


def evidence_matrix(encs_tr, y_tr, encs_te, alpha=1.0):
    """Cross-fit OUTSIDE: caller passes train encodings+labels and test
    encodings; token log-odds estimated on train only."""
    from collections import Counter
    pos, neg, tot = Counter(), Counter(), Counter()
    for e, c in zip(encs_tr, y_tr):
        for t in set(e.ids):
            tot[t] += 1
            pos[t] += c == 1
            neg[t] += c == 0
    ev, lf = {}, {}
    n_all = sum(tot.values())
    for t, n in tot.items():
        ev[t] = np.log((pos[t] + alpha) / (neg[t] + alpha))
        lf[t] = -np.log(n / n_all)
    return np.array([_cell_stats(e.ids, ev, lf, len(ev) + 1000) for e in encs_te],
                    dtype=np.float32)


def evidence_cv(encs, y, n_splits=5, seed=0):
    """Full cross-fit evidence for train rows (for honest train-side eval)."""
    out = np.zeros((len(encs), len(EV_COLS)), dtype=np.float32)
    for tr_i, va_i in StratifiedKFold(n_splits, shuffle=True,
                                      random_state=seed).split(encs, y):
        out[va_i] = evidence_matrix([encs[i] for i in tr_i], y[tr_i],
                                    [encs[i] for i in va_i])
    return out


def design_matrix(df: pd.DataFrame, text_cols: list[str], tok: Tokenizer,
                  ev_tr=None):
    """ev_tr = (encs_tr, y_tr) to build evidence against; None -> sketch only.
    Returns (sketch_df, evidence_df-or-None, encs)."""
    encs = {c: encode(df[c].fillna("").astype(str).tolist(), tok)
            for c in text_cols}
    sk = {f"{c}_sk{i}": None for c in text_cols for i in range(2 * _SKETCH_D)}
    sk_df = pd.DataFrame(
        np.hstack([sketch_matrix(encs[c]) for c in text_cols]),
        columns=list(sk), index=df.index)
    if ev_tr is None:
        return sk_df, None, encs
    ev_parts = [evidence_matrix(ev_tr[0][c], ev_tr[1], encs[c])
                for c in text_cols]
    ev_df = pd.DataFrame(
        np.hstack(ev_parts),
        columns=[f"{c}_{n}" for c in text_cols for n in EV_COLS],
        index=df.index)
    return sk_df, ev_df, encs
