"""Token GIF: per-token evidence bars for one review (train-built log-odds).
Saves figs/lexicon.gif. CPU-only, fast.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from tokenpfn.core import design_matrix, get_tokenizer

df = pd.read_csv("data/Womens Clothing E-Commerce Reviews.csv")
df = df.drop(columns=["Unnamed: 0", "Clothing ID"])
y = df["Recommended IND"].values
Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.25, stratify=y, random_state=0)
tok = get_tokenizer()
_, _, enc_tr = design_matrix(Xtr, ["Review Text"], tok)
pos, neg, tot = Counter(), Counter(), Counter()
for e, c in zip(enc_tr["Review Text"], ytr):
    for t in set(e.ids):
        tot[t] += 1
        pos[t] += c == 1
        neg[t] += c == 0
n_pos, n_neg, V = sum(pos.values()), sum(neg.values()), len(tot)
EV = {t: float(np.log((pos[t] + 1) / (n_pos + V)) - np.log((neg[t] + 1) / (n_neg + V)))
      for t in tot}
row = Xte.iloc[3]
text = row["Review Text"] if isinstance(row["Review Text"], str) else ""
enc = tok.encode(text)
words = [text[s:e] or " " for _, (s, e) in zip(enc.ids, enc.offsets)]
vals = [EV.get(t, 0.0) for t in enc.ids]
take = [i for i in range(len(words)) if words[i].strip()][:14]
words = [words[i][:12] for i in take]
vals = [vals[i] for i in take]

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.set_xlim(-1, 1)
ax.set_ylim(-0.5, len(words) - 0.5)
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words, fontsize=8)
ax.set_xlabel("token log-odds evidence")
ax.set_title("What TabPFN reads in this review")
bars = ax.barh(range(len(words)), [0] * len(words),
               color=["#27ae60" if v > 0 else "#c0392b" for v in vals])


def draw(f):
    n = (f + 1) / 10
    for i, v in enumerate(vals):
        bars[i].set_width(v * n)
    return bars


FuncAnimation(fig, draw, frames=10, interval=350).save(
    "figs/lexicon.gif", writer="pillow", dpi=100)
print("saved figs/lexicon.gif")
