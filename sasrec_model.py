"""SASRec 모델 + 점수 함수 (팀원이 점수 결합 단계에서 import 해서 쓰는 모듈)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SASRec(nn.Module):
    def __init__(self, n_items, d=64, heads=2, layers=2, dropout=0.3, max_len=20):
        super().__init__()
        self.d = d
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=d * 2, dropout=dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def hidden(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.item_emb(x) * (self.d ** 0.5) + self.pos_emb(pos)
        h = self.drop(h)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        return self.ln(self.enc(h, mask=causal))

    def logits(self, h):
        lg = h @ self.item_emb.weight.T
        lg[..., 0] = -1e9
        return lg


class SASRecScorer:
    def __init__(self, model, poi_list, max_len):
        self.model = model.to(DEVICE).eval()
        self.poi_list = poi_list
        self.poi2idx = {p: i for i, p in enumerate(poi_list)}
        self.max_len = max_len
        self.N = len(poi_list)

    @torch.no_grad()
    def score_many(self, histories):
        L = self.max_len
        X = np.zeros((len(histories), L), dtype=np.int64)
        lens = np.ones(len(histories), dtype=np.int64)
        empty = []
        for i, h in enumerate(histories):
            ids = [self.poi2idx[p] + 1 for p in h if p in self.poi2idx][-L:]
            if not ids:
                empty.append(i)
                continue
            X[i, :len(ids)] = ids
            lens[i] = len(ids)
        X, lens = torch.from_numpy(X).to(DEVICE), torch.from_numpy(lens).to(DEVICE)
        hid = self.model.hidden(X)
        last = hid[torch.arange(len(X), device=DEVICE), lens - 1]
        out = F.log_softmax(self.model.logits(last)[:, 1:], dim=-1).cpu().numpy()
        for i in empty:                      # 이력이 없으면 균등 점수
            out[i] = -np.log(self.N)
        return out

    def score(self, history_pois):
        return self.score_many([history_pois])[0]


def load_sasrec(bundle_path):
    b = torch.load(bundle_path, map_location="cpu")
    cfg = b["cfg"]
    model = SASRec(len(b["poi_list"]), d=cfg["d"], layers=cfg["layers"],
                   dropout=cfg["dropout"], max_len=b["max_len"])
    model.load_state_dict(b["state"])
    return SASRecScorer(model, b["poi_list"], b["max_len"])
