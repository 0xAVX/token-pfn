"""A-E experiment (Women's Clothing Reviews, Recommended IND):
A structured-only | B TRANSFORM_TEXT | C sketch | D evidence | E sketch+evidence.
75/25 split, 12k ctx cap. Saves figs/tokenpfn.csv. GPU ~20 min.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tokenpfn.core import design_matrix, evidence_cv, evidence_matrix, get_tokenizer

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TEXT = ["Title", "Review Text"]
STRUCT = ["Age", "Rating", "Positive Feedback Count", "Division Name",
          "Department Name", "Class Name"]


def main():
    t0 = time.time()
    Path("figs").mkdir(exist_ok=True)
    df = pd.read_csv("data/Womens Clothing E-Commerce Reviews.csv")
    df = df.drop(columns=["Unnamed: 0", "Clothing ID"])
    y = df["Recommended IND"].values
    Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.25, stratify=y,
                                          random_state=SEED)
    Xs = Xtr[STRUCT].copy()
    for c in ["Division Name", "Department Name", "Class Name"]:
        Xs[c] = Xs[c].astype("category")
        Xte[c] = Xte[c].astype("category")
    print(f"train={Xs.shape} test={Xte.shape} pos={ytr.mean():.3f}", flush=True)

    from tabpfn import TabPFNClassifier

    def run(Xa, Xb, **kw):
        n = min(12000, len(Xa))
        keep = np.random.RandomState(SEED).choice(len(Xa), n, replace=False)
        m = TabPFNClassifier(random_state=SEED, **kw)
        m.fit(Xa.iloc[keep], ytr[keep])
        return roc_auc_score(yte, m.predict_proba(Xb)[:, 1])

    rows = [("A structured", run(Xs, Xte[STRUCT]))]
    print(f"A: {rows[-1][1]:.4f}", flush=True)

    Xt, Xv = Xtr.copy(), Xte.copy()
    for c in TEXT:
        Xt[c] = Xt[c].astype("string")
        Xv[c] = Xv[c].astype("string")
    rows.append(("B transform-text",
                 run(Xt[STRUCT + TEXT], Xv[STRUCT + TEXT],
                     inference_config={"TRANSFORM_TEXT": True})))
    print(f"B: {rows[-1][1]:.4f}", flush=True)

    tok = get_tokenizer()
    sk_tr, _, encs_tr = design_matrix(Xtr, TEXT, tok)
    sk_te, _, encs_te = design_matrix(Xte, TEXT, tok)
    import tokenpfn.core as T
    cols9 = T.EV_COLS
    evT_tr = evidence_cv(encs_tr["Title"], ytr)
    evB_tr = evidence_cv(encs_tr["Review Text"], ytr)
    evT_te = evidence_matrix(encs_tr["Title"], ytr, encs_te["Title"])
    evB_te = evidence_matrix(encs_tr["Review Text"], ytr, encs_te["Review Text"])
    ev_tr_df = pd.DataFrame(np.hstack([evT_tr, evB_tr]),
                            columns=[f"Title_{c}" for c in cols9] +
                                    [f"Review Text_{c}" for c in cols9],
                            index=Xtr.index)
    ev_te_df = pd.DataFrame(np.hstack([evT_te, evB_te]),
                            columns=ev_tr_df.columns, index=Xte.index)
    sk_tr.index, sk_te.index = Xtr.index, Xte.index
    Xa = pd.concat([Xs, sk_tr], axis=1)
    Xb = pd.concat([Xte[STRUCT], sk_te], axis=1)
    rows.append(("C sketch", run(Xa, Xb)))
    print(f"C: {rows[-1][1]:.4f}", flush=True)
    Xa = pd.concat([Xs, ev_tr_df], axis=1)
    Xb = pd.concat([Xte[STRUCT], ev_te_df], axis=1)
    rows.append(("D evidence", run(Xa, Xb)))
    print(f"D: {rows[-1][1]:.4f}", flush=True)
    Xa = pd.concat([Xs, sk_tr, ev_tr_df], axis=1)
    Xb = pd.concat([Xte[STRUCT], sk_te, ev_te_df], axis=1)
    rows.append(("E sketch+evidence", run(Xa, Xb)))
    print(f"E: {rows[-1][1]:.4f}", flush=True)

    pd.DataFrame(rows, columns=["setup", "auc"]).to_csv(f"figs/tokenpfn_s{SEED}.csv",
                                                         index=False)
    print(f"saved figs/tokenpfn.csv ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
