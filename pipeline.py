"""최종 파이프라인(데모용 재구현).

팀원의 '최종_Test.ipynb'(T3·T4·4단계)와 같은 정의를 한 행(한 사용자)씩 계산하도록 옮겼다.
  1) 시퀀스(SASRec) · 텍스트(ko-sroberta 코사인) · 거리(-log(1+km)) 점수
  2) 행별 z-점수 정규화(방문한 장소 제외) → 가중 결합 (미방문은 시퀀스 항을 빼고 가중치 재정규화)
  3) 결합 점수 상위 50개 후보 → 저·미방문 최소 m개 보장 재랭킹 → 최종 Top-10
설정(final_config.json)은 확정값이므로 기본값으로만 쓰고, 데모에서 바꿔 볼 수는 있다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ["미방문", "저방문", "고방문"]


@dataclass
class Result:
    hist: list[int]                    # 입력 방문 순서 (POI 인덱스)
    weights: dict
    m: int
    sas_top: np.ndarray                # SASRec 단독 Top-10
    before: np.ndarray                 # 결합 점수 Top-10 (재랭킹 전)
    after: np.ndarray                  # 보장 재랭킹 후 Top-10
    added: list[int]                   # 재랭킹으로 새로 들어온 장소
    removed: list[int]                 # 재랭킹으로 빠진 장소
    parts: dict = field(default_factory=dict)   # 항목별 가중 기여도 (전체 POI × 항)
    rel: np.ndarray | None = None      # 결합 점수


class Demo:
    def __init__(self, data_dir: str | Path, load_model: bool = True):
        d = Path(data_dir)
        self.dir = d
        self.poi = pd.read_csv(d / "poi.csv", encoding="utf-8-sig")
        if "spread_km" not in self.poi.columns:
            self.poi["spread_km"] = np.nan
        self.poi_list = self.poi.POI_ID.tolist()
        self.N = len(self.poi_list)
        self.p2i = {p: i for i, p in enumerate(self.poi_list)}
        self.TV = self.poi.train_visits.to_numpy()
        self.GRP = self.poi.group.to_numpy()
        self.X = self.poi.x.to_numpy(dtype="float64")
        self.Y = self.poi.y.to_numpy(dtype="float64")
        self.E = np.load(d / "emb.npy").astype("float32")
        assert self.E.shape[0] == self.N
        self.cfg = json.loads((d / "final_config.json").read_text(encoding="utf-8"))
        self.unseen = (self.TV == 0)
        self.scorer = None
        if load_model:
            self._load_scorer()

    # ---------------------------------------------------------------- 시퀀스 모델
    def _load_scorer(self):
        import sys
        sys.path.insert(0, str(self.dir.parent))
        from sasrec_model import SASRec, SASRecScorer
        import torch
        try:
            b = torch.load(self.dir / "sasrec_bundle.pt", map_location="cpu", weights_only=False)
        except TypeError:                      # 아주 오래된 torch는 weights_only 인자가 없다
            b = torch.load(self.dir / "sasrec_bundle.pt", map_location="cpu")
        cfg = b["cfg"]
        model = SASRec(len(b["poi_list"]), d=cfg["d"], layers=cfg["layers"],
                       dropout=cfg["dropout"], max_len=b["max_len"])
        model.load_state_dict(b["state"])
        assert list(b["poi_list"]) == self.poi_list, "SASRec 후보 목록과 poi.csv가 다르다"
        self.scorer = SASRecScorer(model, b["poi_list"], b["max_len"])

    # ---------------------------------------------------------------- 점수
    def _z(self, v: np.ndarray, hist: list[int]) -> np.ndarray:
        m = np.ones(self.N, bool)
        m[hist] = False
        x = v[m]
        return (v - x.mean()) / (x.std() + 1e-9)

    def component_scores(self, hist: list[int], seq: np.ndarray | None = None) -> dict:
        """원점수 3개: seq(SASRec log-prob), text(코사인), dist(-log(1+km))."""
        if seq is None:
            seq = self.scorer.score_many([[self.poi_list[i] for i in hist]])[0].astype("float32")
        w = np.ones(len(hist))
        w[-1] = 2.0                                           # 마지막 방문 가중 2
        q = (self.E[hist] * w[:, None]).sum(0) / w.sum()
        q = q / (np.linalg.norm(q) + 1e-9)
        text = (q @ self.E.T).astype("float32")
        last = hist[-1]
        x0, y0 = self.X[last], self.Y[last]
        d = np.hypot((self.X - x0) * 88.9, (self.Y - y0) * 111.0)
        dist = -np.log1p(np.where(np.isnan(d), 1e3, d)).astype("float32")
        return {"seq": seq, "text": text, "dist": dist}

    def combine(self, raw: dict, hist: list[int], weights: dict):
        """z-정규화 후 가중 결합. 미방문(Train 방문 0)은 시퀀스 항을 빼고 가중치를 재정규화한다."""
        w = {k: v for k, v in weights.items() if v}
        z = {k: self._z(raw[k].astype("float64"), hist) for k in w}
        avail = {k: (np.where(self.unseen, 0.0, 1.0) if k == "seq" else np.ones(self.N)) for k in w}
        denom = np.maximum(sum(w[k] * avail[k] for k in w), 1e-9)
        parts = {k: w[k] * z[k] * avail[k] / denom for k in w}
        return sum(parts.values()), parts

    def masked_top(self, score: np.ndarray, hist: list[int], k: int) -> np.ndarray:
        s = score.astype("float64").copy()
        s[hist] = -np.inf
        return np.argsort(-s, kind="stable")[:k]

    # ---------------------------------------------------------------- 재랭킹
    def quota(self, cand: np.ndarray, top: int, m: int) -> np.ndarray:
        """상위 top개에 저·미방문이 최소 m개: 관련성이 가장 낮은 고방문을, 후보 중 관련성이 가장 높은 저·미방문으로 교체."""
        KC = len(cand)
        sel = list(range(top))
        low = [j for j in range(KC) if self.GRP[cand[j]] != "고방문"]
        n_low = sum(self.GRP[cand[j]] != "고방문" for j in sel)
        extra = [j for j in low if j not in sel][:max(m - n_low, 0)]
        for j in extra:
            hi = [x for x in sel if self.GRP[cand[x]] == "고방문"]
            if hi:
                sel.remove(hi[-1])
                sel.append(j)
        return cand[sorted(sel)]

    # ---------------------------------------------------------------- 전체 파이프라인
    def recommend(self, hist: list[int], weights: dict | None = None, m: int | None = None,
                  seq: np.ndarray | None = None) -> Result:
        rr = self.cfg["rerank"]
        weights = weights or dict(self.cfg["weights"])
        m = rr["m"] if m is None else m
        raw = self.component_scores(hist, seq)
        rel, parts = self.combine(raw, hist, weights)
        cand = self.masked_top(rel, hist, rr["candidates"])
        top = rr["top"]
        before = cand[:top]
        after = self.quota(cand, top, m)
        return Result(
            hist=list(hist), weights=weights, m=m,
            sas_top=self.masked_top(raw["seq"], hist, top),
            before=before, after=after,
            added=[int(i) for i in after if i not in set(before)],
            removed=[int(i) for i in before if i not in set(after)],
            parts=parts, rel=rel,
        )

    # ---------------------------------------------------------------- 표시용
    def label(self, i: int) -> str:
        r = self.poi.iloc[i]
        return f"{r['name']} ({r['region']})"
