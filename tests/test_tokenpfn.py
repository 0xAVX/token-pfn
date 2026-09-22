import numpy as np
import pandas as pd

from tokenpfn.core import design_matrix, evidence_matrix, get_tokenizer


def test_sketch_and_evidence():
    rng = np.random.RandomState(0)
    tr = pd.DataFrame({"t": ["great product love it"] * 20 + ["terrible broke apart"] * 20})
    y = np.array([1] * 20 + [0] * 20)
    te = pd.DataFrame({"t": ["love this great thing", "terrible awful junk"]})
    tok = get_tokenizer()
    sk_tr, _, enc_tr = design_matrix(tr, ["t"], tok)
    sk_te, _, enc_te = design_matrix(te, ["t"], tok)
    assert sk_tr.shape == (40, 128) and sk_te.shape == (2, 128)
    ev = evidence_matrix(enc_tr["t"], y, enc_te["t"])
    assert ev.shape == (2, 9)
    assert ev[0, 0] > ev[1, 0]  # pos review: stronger positive evidence
