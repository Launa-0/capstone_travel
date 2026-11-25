"""저방문·미방문 관광지 추천 데모 (Streamlit).

실행:  streamlit run app.py
필요:  demo_data/ 폴더 (build_demo_data.py로 만든 것), sasrec_model.py, pipeline.py
"""
import json
import random
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import Demo

HERE = Path(__file__).resolve().parent
DATA = HERE / "demo_data"

st.set_page_config(page_title="저방문·미방문 관광지 추천 데모", page_icon="🧭", layout="wide")

BADGE = {"고방문": "🔵 고방문", "저방문": "🟠 저방문", "미방문": "🟢 미방문"}
COLOR = {"고방문": [52, 120, 201], "저방문": [237, 137, 54], "미방문": [56, 161, 105]}
PART_NAME = {"seq": "시퀀스(SASRec)", "text": "텍스트", "dist": "거리"}


# --------------------------------------------------------------------------- 로딩
@st.cache_resource(show_spinner="모델과 데이터를 불러오는 중…")
def get_demo() -> Demo:
    return Demo(DATA)


@st.cache_data
def get_trips():
    return json.loads((DATA / "trips_val.json").read_text(encoding="utf-8"))


def stretch(fn, *a, **k):
    """Streamlit 버전에 따라 폭 인자가 달라서 둘 다 시도한다."""
    try:
        return fn(*a, width="stretch", **k)
    except TypeError:
        return fn(*a, use_container_width=True, **k)


try:
    demo = get_demo()
except Exception as e:  # noqa: BLE001
    st.error(f"데모 데이터를 불러오지 못했습니다: {e}")
    st.info("`python build_demo_data.py --pre … --step3 … --hand …` 를 먼저 실행해 demo_data/ 폴더를 만드세요. "
            "SASRec을 쓰려면 `pip install torch`도 필요합니다.")
    st.stop()
trips = get_trips()
CFG = demo.cfg
DEF_W = dict(CFG["weights"])
DEF_M = CFG["rerank"]["m"]
PO = demo.poi


def nm(i: int) -> str:
    return str(PO["name"].iloc[i])


# --------------------------------------------------------------------------- 사이드바
st.sidebar.title("🧭 입력")
mode = st.sidebar.radio("입력 방식", ["샘플 여행 (Val)", "직접 입력"], help="샘플은 검증(Val) 데이터의 실제 여행입니다.")

hist: list[int] = []
target: int | None = None
trip_title = ""

if mode == "샘플 여행 (Val)":
    flt = st.sidebar.selectbox(
        "여행 고르기 (탐색용 필터)",
        ["전체", "재랭킹으로 목록이 바뀐 여행", "정답이 저·미방문", "제안 방법만 정답", "SASRec만 정답", "둘 다 못 맞힘"],
        help="데모 탐색을 편하게 하려는 필터입니다. 성능을 보여 주는 근거가 아닙니다.",
    )
    pick = {
        "전체": lambda t: True,
        "재랭킹으로 목록이 바뀐 여행": lambda t: t["changed"],
        "정답이 저·미방문": lambda t: t["tgt_group"] != "고방문",
        "제안 방법만 정답": lambda t: t["final_hit"] and not t["seq_hit"],
        "SASRec만 정답": lambda t: t["seq_hit"] and not t["final_hit"],
        "둘 다 못 맞힘": lambda t: not t["seq_hit"] and not t["final_hit"],
    }[flt]
    pool = [t for t in trips if pick(t) and len(t["visits"]) >= 3]
    if not pool:
        st.sidebar.warning("조건에 맞는 여행이 없습니다.")
        st.stop()
    if st.sidebar.button("🎲 무작위 여행"):
        st.session_state.trip_id = random.choice(pool)["id"]
    ids = [t["id"] for t in pool]
    if st.session_state.get("trip_id") not in ids:
        sp_ = PO["spread_km"].fillna(0).to_numpy()
        ok_ = [t for t in pool if len(t["visits"]) >= 4 and t["sas_low"] <= 3 and t["changed"]
               and all(sp_[demo.p2i[p]] <= 30 for p in t["visits"])]
        st.session_state.trip_id = (ok_ or pool)[0]["id"]

    def fmt(tid):
        t = next(x for x in pool if x["id"] == tid)
        first = nm(demo.p2i[t["visits"][0]])
        return f"{tid} · {first} 외 {len(t['visits']) - 1}곳"

    tid = st.sidebar.selectbox("여행", ids, key="trip_id", format_func=fmt)
    trip = next(t for t in pool if t["id"] == tid)
    vis = [demo.p2i[p] for p in trip["visits"]]
    k = st.sidebar.slider("몇 번째 방문까지 입력했다고 볼까요?", 1, len(vis) - 1, len(vis) - 1,
                          help="이 지점 다음 방문지가 '정답'입니다. 슬라이더를 줄이면 여행 초반 상황을 볼 수 있습니다.")
    hist, target = vis[:k], vis[k]
    trip_title = f"여행 {tid} ({trip['year']}년, 관광지 {len(vis)}곳 방문)"
else:
    opts = list(range(demo.N))
    sel = st.sidebar.multiselect(
        "방문한 장소를 방문한 순서대로 고르세요", opts,
        format_func=lambda i: f"{nm(i)} · {PO['region'].iloc[i]}",
        placeholder="장소 이름을 검색 (예: 오동도)",
        help="고른 순서가 방문 순서입니다. 마지막으로 고른 장소가 '직전 방문지'가 됩니다.",
    )
    hist = list(sel)
    trip_title = "직접 입력한 방문 순서"
    if not hist:
        st.title("저방문·미방문 관광지 추천 데모")
        st.info("왼쪽에서 방문한 장소를 하나 이상 골라 주세요.")
        st.stop()

with st.sidebar.expander("⚙️ 파이프라인 설정 (데모용)"):
    st.caption("기본값은 팀이 확정한 설정입니다. 바꾸면 포스터·결과표의 수치와 대응하지 않습니다.")
    if st.button("확정값으로 되돌리기"):
        for key, v in {"w_seq": DEF_W["seq"], "w_text": DEF_W["text"], "w_dist": DEF_W["dist"], "m": DEF_M}.items():
            st.session_state[key] = v
    w_seq = st.slider("시퀀스 가중치", 0.0, 2.0, float(DEF_W["seq"]), 0.05, key="w_seq")
    w_text = st.slider("텍스트 가중치", 0.0, 2.0, float(DEF_W["text"]), 0.05, key="w_text")
    w_dist = st.slider("거리 가중치", 0.0, 2.0, float(DEF_W["dist"]), 0.05, key="w_dist")
    m = st.slider("저·미방문 최소 보장 수 m (0이면 재랭킹 없음)", 0, 4, DEF_M, key="m")
weights = {"seq": w_seq, "text": w_text, "dist": w_dist}
if sum(weights.values()) == 0:
    st.sidebar.error("가중치가 모두 0입니다.")
    st.stop()
custom = (weights != DEF_W) or (m != DEF_M)

# --------------------------------------------------------------------------- 계산
res = demo.recommend(hist, weights=weights, m=m)
TV, GRP = demo.TV, demo.GRP

# --------------------------------------------------------------------------- 본문
st.title("저방문·미방문 관광지 추천 데모")
st.caption("AI-Hub 국내여행로그(서부권) · 시퀀스(SASRec) + 텍스트 + 거리 결합 → 저·미방문 최소 보장 재랭킹 · 후보 풀: 관광지 3,389곳")
if custom:
    st.warning("확정 설정과 다른 값으로 돌리고 있습니다. 이 화면의 결과는 탐색용이며 포스터·결과표의 수치와 대응하지 않습니다.")

st.subheader("1. 입력 — 지금까지의 방문 순서")
chips = "  →  ".join(f"**{n + 1}. {nm(i)}** ({PO['region'].iloc[i]})" for n, i in enumerate(hist))
st.markdown(chips)
st.caption(f"{trip_title}")
if target is not None:
    st.markdown(f"**실제 다음 방문지(정답)**: ⭐ {nm(target)} ({PO['region'].iloc[target]}) · "
                f"{BADGE[GRP[target]]} · 학습 데이터 방문 {int(TV[target])}회")


SPREAD = PO["spread_km"].fillna(0).to_numpy()


def sgg(i: int) -> str:
    toks = str(PO["region"].iloc[i]).split()
    return toks[-1] if toks else ""


def table(lst, kind):
    rows, tags = [], []
    for r, i in enumerate(lst, 1):
        tag = []
        if target is not None and i == target:
            tag.append("⭐")
        if kind == "after" and i in res.added:
            tag.append("🆕")
        if kind == "before" and i in res.removed:
            tag.append("⬇")
        if SPREAD[i] > 30:
            tag.append("⚠")
        tags.append(" ".join(tag))
        rows.append({"#": r, "장소": f"{' '.join(tag)} {nm(i)} · {sgg(i)}".strip(),
                     "방문": f"{BADGE[GRP[i]][0]} {int(TV[i])}"})
    df = pd.DataFrame(rows)

    def color(row):
        t = tags[row.name]
        c = "background-color:#fff3b0" if "⭐" in t else "background-color:#dff5df" if "🆕" in t \
            else "background-color:#fde8e8" if "⬇" in t else ""
        return [c] * len(row)

    return df.style.apply(color, axis=1)


def stats(lst):
    low = int((GRP[lst] != "고방문").sum())
    return low, float(TV[lst].mean())


st.subheader("2. 추천 결과 — Top-10")
cols = st.columns(3)
panels = [
    ("SASRec 단독", res.sas_top, "none", "방문 순서만 보는 기준 모델"),
    ("결합 (재랭킹 전)", res.before, "before", "시퀀스 + 텍스트 + 거리"),
    (f"제안 (재랭킹 후, 보장 m={m})" if m else "제안 (재랭킹 없음)", res.after, "after", "저·미방문 최소 보장 적용"),
]
base_low, base_arp = stats(res.sas_top)
for col, (title, lst, kind, sub) in zip(cols, panels):
    with col:
        st.markdown(f"#### {title}")
        st.caption(sub)
        low, arp = stats(lst)
        a, b = st.columns(2)
        a.metric("저·미방문 추천", f"{low} / 10", delta=None if kind == "none" else f"{low - base_low:+d} (SASRec 대비)")
        b.metric("평균 학습 방문수", f"{arp:.1f}", delta=None if kind == "none" else f"{arp - base_arp:+.1f}",
                 delta_color="inverse")
        st.dataframe(table(lst, kind), hide_index=True, height=392,
                     column_config={"#": st.column_config.NumberColumn(width="small"), "방문": st.column_config.TextColumn(width="small")})
        if target is not None:
            st.markdown("정답 포함: " + ("✅ **적중**" if target in set(lst) else "❌ 없음"))

st.info("**읽는 법** · 방문 열: 🔵 고방문(학습 방문 6회 이상) 🟠 저방문(1~5회) 🟢 미방문(0회) 뒤의 숫자는 학습 데이터 방문 횟수 · "
        "⭐ 정답 · 🆕 보장으로 추가 · ⬇ 재랭킹에서 제외 · ⚠ 방문 좌표가 30km 넘게 흩어진 장소(원본 데이터의 장소 매칭 오류 가능).  \n"
        "한 여행의 정답 적중 여부는 사례일 뿐이며 성능 근거가 아닙니다. 전체 성능은 '실험 결과' 탭을 보세요.")

# --------------------------------------------------------------------------- 탭
t1, t2, t3, t4 = st.tabs(["🔁 재랭킹 전후", "🧮 점수 구성", "🗺️ 지도", "📊 실험 결과"])

with t1:
    n_before = stats(res.before)[0]
    n_after = stats(res.after)[0]
    st.markdown(
        f"결합 점수 상위 50개 후보 중 Top-10을 고릅니다. 결합 Top-10에는 저·미방문이 **{n_before}개** 있었고, "
        f"보장 규칙(최소 {m}개) 적용 후 **{n_after}개**가 됩니다." if m else "보장 수가 0이라 재랭킹을 적용하지 않습니다.")
    if res.added:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🆕 새로 들어온 장소** (후보 50개 중 관련성이 가장 높은 저·미방문)")
            for i in res.added:
                st.markdown(f"- {nm(i)} · {BADGE[GRP[i]]} (학습 방문 {int(TV[i])}회) — 결합 순위 {int(np.where(demo.masked_top(res.rel, hist, 50) == i)[0][0]) + 1}위")
        with c2:
            st.markdown("**⬇ 빠진 장소** (Top-10 중 관련성이 가장 낮은 고방문)")
            for i in res.removed:
                st.markdown(f"- {nm(i)} · {BADGE[GRP[i]]} (학습 방문 {int(TV[i])}회)")
    elif m:
        st.success("이미 저·미방문이 충분해서 바뀐 장소가 없습니다.")

    ch = []
    for title, lst in (("SASRec 단독", res.sas_top), ("결합(전)", res.before), ("제안(후)", res.after)):
        for r, i in enumerate(lst, 1):
            ch.append({"방법": title, "순위": r, "학습 방문수": int(TV[i]), "장소": nm(i)})
    cdf = pd.DataFrame(ch)
    chart = (alt.Chart(cdf).mark_bar().encode(
        x=alt.X("순위:O", title="추천 순위"),
        xOffset="방법:N",
        y=alt.Y("학습 방문수:Q", scale=alt.Scale(type="symlog"), title="학습 방문수 (로그 눈금)"),
        color=alt.Color("방법:N", scale=alt.Scale(domain=["SASRec 단독", "결합(전)", "제안(후)"],
                                                 range=["#9aa5b1", "#3478c9", "#ed8936"])),
        tooltip=["방법", "순위", "장소", "학습 방문수"],
    ).properties(height=320, title="추천된 장소의 학습 방문수 (낮을수록 덜 알려진 곳)"))
    stretch(st.altair_chart, chart)

with t2:
    st.markdown("각 장소의 결합 점수는 **행별 z-점수**로 맞춘 세 점수의 가중합입니다. "
                "방문 기록이 없는 장소(미방문)는 시퀀스 항을 빼고 나머지 가중치를 다시 맞춥니다.")
    which = st.radio("볼 목록", ["제안 (재랭킹 후)", "결합 (재랭킹 전)"], horizontal=True)
    lst = res.after if which.startswith("제안") else res.before
    rows = []
    for r, i in enumerate(lst, 1):
        lab = f"{r}. {nm(i)}"
        for kx, v in res.parts.items():
            rows.append({"장소": lab, "점수 항": PART_NAME[kx], "기여도": float(v[i]), "순위": r})
    pdf = pd.DataFrame(rows)
    order = [f"{r}. {nm(i)}" for r, i in enumerate(lst, 1)]
    ch2 = (alt.Chart(pdf).mark_bar().encode(
        y=alt.Y("장소:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)),
        x=alt.X("기여도:Q", title="가중 z-점수 기여도"),
        color=alt.Color("점수 항:N", scale=alt.Scale(domain=list(PART_NAME.values()),
                                                   range=["#3478c9", "#38a169", "#d69e2e"])),
        tooltip=["장소", "점수 항", alt.Tooltip("기여도:Q", format=".2f")],
    ).properties(height=380))
    stretch(st.altair_chart, ch2)
    if which.startswith("제안") and res.added:
        st.caption("🆕 보장으로 들어온 장소는 결합 점수만으로는 Top-10 밖이었던 곳이라, 위 합계가 다른 장소보다 낮을 수 있습니다.")
        st.caption("보장으로 추가: " + ", ".join(nm(i) for i in res.added))

with t3:
    import pydeck as pdk

    def row(i, kind, r=None):
        return {"name": nm(i), "lon": float(demo.X[i]), "lat": float(demo.Y[i]), "kind": kind,
                "info": f"{kind} · {GRP[i]} (학습 {int(TV[i])}회)" + (f" · 추천 {r}위" if r else "")}

    pts = []
    for n, i in enumerate(hist, 1):
        pts.append({**row(i, f"입력 {n}번째"), "color": [90, 90, 90], "radius": 450})
    for r, i in enumerate(res.after, 1):
        pts.append({**row(i, "제안 추천", r), "color": COLOR[GRP[i]], "radius": 650})
    for i in res.removed:
        pts.append({**row(i, "재랭킹에서 제외"), "color": [190, 190, 190], "radius": 400})
    if target is not None:
        pts.append({**row(target, "정답"), "color": [255, 215, 0], "radius": 1100})
    pdf2 = pd.DataFrame(pts).dropna(subset=["lon", "lat"])
    path = [[float(demo.X[i]), float(demo.Y[i])] for i in hist if not np.isnan(demo.X[i])]
    layers = [
        pdk.Layer("PathLayer", data=[{"path": path}], get_path="path", get_color=[90, 90, 90], width_min_pixels=2),
        pdk.Layer("ScatterplotLayer", data=pdf2[pdf2.kind == "정답"], get_position=["lon", "lat"], get_radius="radius",
                  get_fill_color="color", radius_min_pixels=14, pickable=True),
        pdk.Layer("ScatterplotLayer", data=pdf2[pdf2.kind != "정답"], get_position=["lon", "lat"], get_radius="radius",
                  get_fill_color="color", radius_min_pixels=6, pickable=True, stroked=True,
                  get_line_color=[255, 255, 255], line_width_min_pixels=1),
        pdk.Layer("TextLayer", data=pdf2[pdf2.kind.str.startswith("입력")], get_position=["lon", "lat"], get_text="name",
                  get_size=13, get_color=[40, 40, 40], get_pixel_offset=[0, -16]),
    ]
    lat_c, lon_c = pdf2.lat.mean(), pdf2.lon.mean()
    span = max(pdf2.lat.max() - pdf2.lat.min(), (pdf2.lon.max() - pdf2.lon.min()) * 0.8, 0.05)
    zoom = float(np.clip(np.log2(180 / span) - 1.2, 6, 12))
    deck = pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom),
                    tooltip={"text": "{name}\n{info}"}, map_provider="carto", map_style="light")
    st.pydeck_chart(deck, height=520)
    st.caption("회색 선: 입력한 방문 순서 · 색 점: 제안 추천(🔵고방문 🟠저방문 🟢미방문) · 노란 점: 정답 · 연회색 점: 재랭킹에서 빠진 장소")

with t4:
    st.markdown("**평가용(Test) 373건 결과** — 포스터 표 1과 같은 수치입니다. (설정을 모두 확정한 뒤 Test를 한 번만 열었습니다.)")
    try:
        tt = pd.read_csv(DATA / "results_test.csv", encoding="utf-8-sig", index_col=0)
        want = {"SASRec": "SASRec 단독", "지역+인기(기준선)": "지역+인기 (사후 기준선)", "CF": "협업 필터링(CF)",
                "최종 (시퀀스+텍스트+거리)": "결합 (재랭킹 전)", "4단계 최종 (보장 m=2)": "제안 (보장 m=2)"}
        sub = tt.loc[list(want)].rename(index=want)
        sub = sub[["전체", "저방문", "고방문", "미방문", "저·미방문 노출비율", "ARP(평균 Train 방문수)", "Coverage"]]
        sub.columns = ["Recall@10 전체", "저방문", "고방문", "미방문", "저·미방문 노출비율", "평균 학습 방문수(ARP)", "Coverage"]
        st.dataframe(sub.style.format("{:.3f}").format("{:.1f}", subset=["평균 학습 방문수(ARP)"]))
    except Exception as e:  # noqa: BLE001
        st.warning(f"결과표를 읽지 못했습니다: {e}")
    st.caption("미방문 열은 Test의 미방문 정답이 2건뿐이라 해석하지 않습니다.")
    st.markdown(
        "- 전체 Recall@10은 SASRec 0.416 → 제안 0.389로 낮았지만, 차이는 통계적으로 유의하지 않았습니다(−0.027, 95% 구간 [−0.056, +0.003]). **정확도가 유지됐다고 단정할 수는 없습니다.**\n"
        "- 저·미방문 추천 비율은 27.9% → 39.4%로, 평균 학습 방문수는 21.0 → 17.7로 바뀌었습니다.\n"
        "- 지역+인기 기준선은 결과를 확인한 뒤 추가한 사후 비교입니다.\n"
        "- 검증(Val)에서 보인 텍스트·거리 결합의 이득은 Test에서 재현되지 않았습니다. 미방문 정답은 Test에 2건뿐이라 미방문 성능은 평가하지 못했고, 정답의 17.2%는 후보 풀 밖이라 어떤 모델도 맞힐 수 없었습니다."
    )
