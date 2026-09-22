"""PFN Lexicon demo: review -> highlighted token evidence + TabPFN verdict.
Fits once at startup (~1 min), then instant. Run: <venv-python> demo/app.py (5005)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from tokenpfn.core import design_matrix, evidence_matrix, get_tokenizer

STRUCT = ["Age", "Rating", "Positive Feedback Count", "Division Name",
          "Department Name", "Class Name"]
TEXT = ["Title", "Review Text"]
COLS9 = ["ev_pos_max", "ev_neg_min", "ev_mean", "ev_top3", "ev_var",
         "ev_rarity", "ev_unk", "ev_len", "ev_lexdiv"]

print("loading...", flush=True)
df = pd.read_csv("data/Womens Clothing E-Commerce Reviews.csv")
df = df.drop(columns=["Unnamed: 0", "Clothing ID"])
y = df["Recommended IND"].values
Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.25, stratify=y, random_state=0)
Xs = Xtr[STRUCT].copy()
for c in ["Division Name", "Department Name", "Class Name"]:
    Xs[c] = Xs[c].astype("category")
keep = np.random.RandomState(0).choice(len(Xs), 8000, replace=False)
tok = get_tokenizer()
_, _, enc_tr = design_matrix(Xtr, TEXT, tok)
ev_all, lf_all = {}, {}
from collections import Counter
pos, neg, tot = Counter(), Counter(), Counter()
for col in TEXT:
    for e, c in zip(enc_tr[col], ytr):
        for t in set(e.ids):
            tot[t] += 1
            pos[t] += c == 1
            neg[t] += c == 0
EV = {t: float(np.log((pos[t] + 1) / (neg[t] + 1))) for t in tot}
m = TabPFNClassifier(random_state=0)
m.fit(Xs.iloc[keep], ytr[keep])
SAMPLES = Xte.iloc[np.random.RandomState(1).choice(len(Xte), 12,
                                                   replace=False)].index.tolist()
print("ready.", flush=True)

app = Flask(__name__)
PAGE = """
<h1>PFN Lexicon — what did TabPFN read?</h1>
<form method=get>Case: <select name=i>{% for k in ids %}<option {{'selected' if k==i}}>{{k}}</option>{% endfor %}</select>
<input type=submit value="Show"></form>
<h2>Would recommend? {{'%.0f' % (100*p)}}%</h2>
<p>{{hl|safe}}</p>
<p><small>green = positive evidence, red = negative (per-token log-odds)</small></p>
"""


def highlight(text, ev):
    enc = tok.encode(text if isinstance(text, str) else "")
    out = []
    for t, (s, e) in zip(enc.ids, enc.offsets):
        v = ev.get(t, 0.0)
        col = "#27ae60" if v > 0.15 else ("#c0392b" if v < -0.15 else "#888")
        word = text[s:e] or " "
        out.append(f'<span style="background:{col}22;border-bottom:2px solid {col}"'
                   f' title="{v:.2f}">{word}</span>')
    return "".join(out) or "(empty)"


@app.get("/")
def index():
    i = int(request.args.get("i", SAMPLES[0]))
    row = Xte.loc[i]
    q = row[STRUCT].to_frame().T.copy()
    for c in ["Division Name", "Department Name", "Class Name"]:
        q[c] = q[c].astype("category")
    p = float(m.predict_proba(q)[:, 1][0])
    hl = "<br><br>".join(highlight(row[c], EV) for c in TEXT)
    return render_template_string(PAGE, ids=SAMPLES, i=i, p=p, hl=hl)


if __name__ == "__main__":
    app.run(debug=True, port=5005)
