"""데모용 데이터 폴더(demo_data/) 만들기. 한 번만 실행하면 된다.

사용법
  python build_demo_data.py --pre <preprocessed_combined 폴더> --step3 <step3_out 폴더> --hand <sasrec_handoff 폴더>

만드는 것
  demo_data/poi.csv              후보 풀 3,389곳 (이름·지역·유형·Train 방문수·그룹·좌표)
  demo_data/emb.npy              ko-sroberta 장소 임베딩
  demo_data/sasrec_bundle.pt     SASRec 가중치
  demo_data/final_config.json    확정 설정 (가중치·재랭킹)
  demo_data/trips_val.json       샘플 여행 (Val) + 각 방법이 정답을 맞혔는지 미리 계산한 표시
  demo_data/results_*.csv        결과표 (포스터와 같은 수치)
끝에 Val 재현 검사(0.404 / 0.402 등)를 돌려서 팀원 결과와 일치하는지 확인한다.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import Demo, GROUPS  # noqa: E402

TOUR = [1, 2, 3, 6, 7, 13]


def merge_consecutive(df):
    df = df.sort_values(["TRAVEL_ID", "VISIT_ORDER"]).copy()
    prev = df.groupby("TRAVEL_ID")["POI_ID"].shift(1)
    out = df[~(df["POI_ID"].notna() & (df["POI_ID"] == prev))].copy()
    out["VISIT_ORDER"] = out.groupby("TRAVEL_ID").cumcount() + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--step3", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--out", default=str(HERE / "demo_data"))
    a = ap.parse_args()
    pre, s3, hand, out = Path(a.pre), Path(a.step3), Path(a.hand), Path(a.out)
    out.mkdir(exist_ok=True)

    # --- 장소 표 -----------------------------------------------------------
    cd = pd.read_csv(s3 / "poi_docs.csv", encoding="utf-8-sig")
    parts = cd.doc.fillna("").str.split(" | ", regex=False)
    cd["name"] = parts.str[0]
    cd["region"] = parts.str[1].fillna("")
    cd["type"] = parts.str[2].where(parts.str.len() >= 3, "").fillna("")
    cd.loc[cd["type"].str.startswith("활동:"), "type"] = ""

    def load(split):
        d = pd.read_csv(pre / split / "visit_area_info.csv", encoding="utf-8-sig", low_memory=False)
        d["POI_ID"] = d.POI_ID.str.strip()
        return d

    vi = {s: load(s) for s in ("train", "val")}
    vd = {s: merge_consecutive(vi[s]) for s in vi}
    tour = lambda v: v[v.VISIT_AREA_TYPE_CD.isin(TOUR)]
    vv = pd.concat([tour(vd["train"]), tour(vd["val"])])
    xy = (vv.assign(X=pd.to_numeric(vv.X_COORD, errors="coerce"), Y=pd.to_numeric(vv.Y_COORD, errors="coerce"))
          .groupby("POI_ID")[["X", "Y"]].median().reindex(cd.POI_ID))
    cd["x"], cd["y"] = xy.X.values, xy.Y.values
    # 같은 POI_ID의 방문 좌표가 서로 멀리 흩어져 있으면 POI 매칭 오류 가능성이 있다 (데모에서 ⚠로 표시)
    vc = vv.assign(X=pd.to_numeric(vv.X_COORD, errors="coerce"), Y=pd.to_numeric(vv.Y_COORD, errors="coerce")).dropna(subset=["X", "Y"])
    med = vc.groupby("POI_ID")[["X", "Y"]].transform("median")
    vc = vc.assign(km=np.hypot((vc.X - med.X) * 88.9, (vc.Y - med.Y) * 111.0))
    cnt = vc.groupby("POI_ID").size()
    sp = vc.groupby("POI_ID").km.max().where(cnt >= 2)
    cd["spread_km"] = sp.reindex(cd.POI_ID).values
    assert cd.POI_ID.tolist() == sorted(set(tour(vd["train"]).POI_ID.dropna()) | set(tour(vd["val"]).POI_ID.dropna())), \
        "후보 풀(Train∪Val)이 팀원 정의와 다르다"
    cd[["POI_ID", "name", "region", "type", "train_visits", "group", "doc", "x", "y", "spread_km"]].to_csv(
        out / "poi.csv", index=False, encoding="utf-8-sig")

    # --- 복사 ---------------------------------------------------------------
    cfg = json.loads((s3 / "final_config.json").read_text(encoding="utf-8"))
    shutil.copy(s3 / f"poi_text_emb_{cfg['embedding']}.npy", out / "emb.npy")
    shutil.copy(hand / "sasrec_bundle.pt", out / "sasrec_bundle.pt")
    shutil.copy(s3 / "final_config.json", out / "final_config.json")
    for f in ("results_test.csv", "results_test_paired.csv", "results_val.csv", "results_val_paired.csv"):
        if (s3 / f).exists():
            shutil.copy(s3 / f, out / f)
    if (s3 / "rerank_val_grid.csv").exists():
        shutil.copy(s3 / "rerank_val_grid.csv", out / "rerank_val_grid.csv")
    (out / "figs").mkdir(exist_ok=True)
    for f in ("pipeline_final.png", "rerank_trade_off.png", "test_tradeoff_group.png", "test_paired_forest.png"):
        if (s3 / f).exists():
            shutil.copy(s3 / f, out / "figs" / f)
    shutil.copy(hand / "sasrec_model.py", HERE / "sasrec_model.py")

    # --- 샘플 여행 (Val) -----------------------------------------------------
    demo = Demo(out)
    p2i = demo.p2i
    keep = vd["val"][vd["val"].VISIT_AREA_TYPE_CD.isin(TOUR) & vd["val"].POI_ID.isin(p2i)].sort_values(["TRAVEL_ID", "VISIT_ORDER"])
    trips = []
    for tid, g in keep.groupby("TRAVEL_ID"):
        if len(g) >= 2:
            trips.append({"id": tid, "year": int(g.YEAR.iloc[0]), "visits": g.POI_ID.tolist()})
    ref = pd.read_csv(hand / "val_eval_rows.csv", encoding="utf-8-sig")
    assert [t["id"] for t in trips] == list(ref.TRAVEL_ID), "Val 평가 행(순서)이 팀원 저장본과 다르다"
    assert [t["visits"][-1] for t in trips] == list(ref.target_poi)

    # --- 전체 Val을 돌려 팀원 수치와 맞는지 확인하고, 방법별 적중 표시를 저장 ---------------
    hist = [[p2i[p] for p in t["visits"][:-1]] for t in trips]
    seq = demo.scorer.score_many([[demo.poi_list[i] for i in h] for h in hist]).astype("float32")
    saved = np.load(hand / "sasrec_val_scores.npy")
    top10 = lambda M, r: demo.masked_top(M[r], hist[r], 10)
    agree = np.mean([set(top10(seq, r)) == set(top10(saved, r)) for r in range(len(trips))])
    print(f"SASRec 점수 vs 저장본: 상위10 목록 일치율 {agree:.3f}")

    rows = []
    for r, t in enumerate(trips):
        res = demo.recommend(hist[r], seq=seq[r])
        tgt = p2i[t["visits"][-1]]
        t["tgt_group"] = str(demo.GRP[tgt])
        t["seq_hit"] = bool(tgt in set(res.sas_top))
        t["comb_hit"] = bool(tgt in set(res.before))
        t["final_hit"] = bool(tgt in set(res.after))
        t["sas_low"] = int((demo.GRP[res.sas_top] != "고방문").sum())      # SASRec Top-10의 저·미방문 수
        t["changed"] = bool(set(res.before) != set(res.after))             # 재랭킹으로 목록이 바뀌었는가
        rows.append((t["tgt_group"], t["seq_hit"], t["comb_hit"], t["final_hit"],
                     np.mean(demo.GRP[res.before] != "고방문"), np.mean(demo.GRP[res.after] != "고방문")))
    df = pd.DataFrame(rows, columns=["g", "seq", "comb", "final", "exp_b", "exp_a"])
    summ = lambda c: {**{g: round(df[df.g == g][c].mean(), 3) for g in GROUPS}, "전체": round(df[c].mean(), 3)}
    print("\n[Val 재현 검사] 기대값: SASRec 전체 0.364 / 결합 0.404·0.194·0.544 / 4단계 0.402·0.216·0.532 / 노출 0.377→0.425")
    for c, nm in (("seq", "SASRec"), ("comb", "결합(재랭킹 전)"), ("final", "4단계 최종")):
        print(f"  {nm:12s}", summ(c))
    print(f"  저·미방문 노출 전 {df.exp_b.mean():.3f} → 후 {df.exp_a.mean():.3f}")
    (out / "trips_val.json").write_text(json.dumps(trips, ensure_ascii=False), encoding="utf-8")
    print(f"\n완료: {out} (샘플 여행 {len(trips)}개, 장소 {demo.N}개)")


if __name__ == "__main__":
    main()
