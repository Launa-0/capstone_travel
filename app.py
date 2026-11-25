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
from study import render_study  # noqa: E402

st.sidebar.divider()
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
    m = st.slider("저·미방문 최소 보장 수 m (0이면 재랭킹 없음)", 0, 3, DEF_M, key="m", help="보고서에서 평가한 값은 1~3입니다.")
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


def reason(i: int) -> str:
    if i in res.added:
        return "저·미방문 최소 보장으로 추가"
    pos = {k: max(float(v[i]), 0.0) for k, v in res.parts.items()}
    tot = sum(pos.values()) or 1.0
    name = {"seq": "방문 순서", "text": "텍스트 유사", "dist": "가까운 거리"}
    top = sorted(pos, key=pos.get, reverse=True)[:2]
    return " · ".join(f"{name[k]} {pos[k] / tot:.0%}" for k in top if pos[k] > 0) or "-"


def km_from_last(i: int) -> float:
    last = hist[-1]
    d = np.hypot((demo.X[i] - demo.X[last]) * 88.9, (demo.Y[i] - demo.Y[last]) * 111.0)
    return float(d)


def result_table(lst):
    rows, tags = [], []
    for r, i in enumerate(lst, 1):
        tag = []
        if target is not None and i == target:
            tag.append("⭐")
        if i in res.added:
            tag.append("🆕")
        if SPREAD[i] > 30:
            tag.append("⚠")
        tags.append(" ".join(tag))
        d = km_from_last(i)
        rows.append({"#": r, "장소": f"{' '.join(tag)} {nm(i)}".strip(), "지역": PO["region"].iloc[i],
                     "구분": f"{BADGE[GRP[i]]} ({int(TV[i])}회)", "직전 방문지에서": ("-" if np.isnan(d) else f"{d:.1f}km"),
                     "추천 이유 (점수 기여)": reason(i)})
    df = pd.DataFrame(rows)

    def color(row):
        t = tags[row.name]
        c = "background-color:#fff3b0" if "⭐" in t else "background-color:#dff5df" if "🆕" in t else ""
        return [c] * len(row)

    return df.style.apply(color, axis=1)


st.subheader("2. 추천 결과 — Top-10")
low_n, arp_v = stats(res.after)
c1, c2, c3 = st.columns([1, 1, 2])
c1.metric("덜 알려진 곳(저·미방문)", f"{low_n} / 10")
c2.metric("평균 학습 방문수", f"{arp_v:.1f}")
if target is not None:
    c3.markdown("**실제 다음 방문지 포함 여부**  \n" + ("✅ **적중**" if target in set(res.after) else "❌ 이번 사례에서는 없음"))
st.dataframe(result_table(res.after), hide_index=True, height=392,
             column_config={"#": st.column_config.NumberColumn(width="small")})

st.info("**읽는 법** · 구분: 🔵 고방문(학습 방문 6회 이상) 🟠 저방문(1~5회) 🟢 미방문(0회), 괄호는 학습 데이터 방문 횟수 · "
        "🆕 저·미방문 보장으로 들어온 장소 · ⭐ 실제 다음 방문지 · ⚠ 방문 좌표가 30km 넘게 흩어진 장소(원본 데이터의 장소 매칭 오류 가능).  \n"
        "한 여행의 적중 여부는 사례일 뿐이며 성능 근거가 아닙니다. 전체 성능은 '실험 결과' 탭을 보세요.")

# --------------------------------------------------------------------------- 탭
t1, t2, t3, t4, t5 = st.tabs(["🔁 방식 비교 (재랭킹 전후)", "🧮 점수 구성", "🗺️ 지도", "📊 실험 결과", "🔬 왜 이렇게 만들었나"])

with t1:
    st.caption("같은 입력으로 방식을 바꿔 본 결과입니다 (발표·분석용). 메인 화면의 추천은 오른쪽 '제안' 한 가지입니다.")
    def _stats_tbl(lst, kind):
        return table(lst, kind)
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
            st.dataframe(_stats_tbl(lst, kind), hide_index=True, height=392,
                         column_config={"#": st.column_config.NumberColumn(width="small"), "방문": st.column_config.TextColumn(width="small")})
            if target is not None:
                st.markdown("정답 포함: " + ("✅ **적중**" if target in set(lst) else "❌ 없음"))


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
    st.caption("평가용(Test) 373건 · 설정을 모두 확정한 뒤 Test를 한 번만 열어 얻은 결과입니다. 해석과 판정은 '📚 연구 보고서'의 3·4단계를 보세요.")
    k1, k2, k3, k4 = st.tabs(["핵심 요약", "Test 전체 표·차이", "Val 설정 선택", "그림"])
    tt = pd.read_csv(DATA / "results_test.csv", encoding="utf-8-sig", index_col=0)
    pr = pd.read_csv(DATA / "results_test_paired.csv", encoding="utf-8-sig", index_col=0)

    with k1:
        want = {"SASRec": "SASRec 단독", "지역+인기(기준선)": "지역+인기 (기준선)", "CF": "협업 필터링(CF)",
                "최종 (시퀀스+텍스트+거리)": "결합 (재랭킹 전)", "4단계 최종 (보장 m=2)": "제안 (보장 m=2)"}
        sub = tt.loc[list(want)].rename(index=want)[["전체", "저방문", "고방문", "저·미방문 노출비율", "ARP(평균 Train 방문수)", "Coverage"]]
        sub.columns = ["Recall@10 전체", "저방문", "고방문", "저·미방문 노출비율", "평균 학습 방문수(ARP)", "Coverage"]
        st.dataframe(sub.style.format("{:.3f}").format("{:.1f}", subset=["평균 학습 방문수(ARP)"]))
        st.caption("미방문 정답은 Test에 2건뿐이라 미방문 열은 표에서 뺐습니다.")
        st.markdown(
            "**쓸 수 있는 주장**\n"
            "- 보장 재랭킹으로 저·미방문 노출이 늘었습니다: 노출비율 0.279 → 0.345(결합) → 0.394(제안), Coverage 0.270 → 0.321, 평균 학습 방문수 21.0 → 17.7. 다만 보장 규칙이 만든 값이라 예측력의 증거는 아닙니다.\n"
            "- 같은 전체 Recall에서 지역+인기 기준선보다 저방문 Recall이 높습니다: **+0.099 [+0.033, +0.176] ★** (0.099 → 0.198).\n"
            "- 전체 Recall 차이(제안 − SASRec)는 −0.027 [−0.056, +0.003]으로 통계적으로 유의하지 않았습니다. 다만 **유지가 입증된 것은 아니며** “손실이 없다”고 쓰면 안 됩니다.\n\n"
            "**주장하지 않는 것**\n"
            "- 결합 모델이 SASRec보다 정확하다 · 텍스트·거리가 Recall을 높인다 (Val의 이득이 Test에서 재현되지 않음)\n"
            "- 미방문을 개선했다 (Test 미방문 n=2)\n\n"
            "**한계**: 선택과 평가가 같은 Val이라 Val 수치는 낙관적 · 정답의 17.2%(64행)는 후보 풀 밖, 8.3%(31행)는 풀 안 입력 이력이 비어 모든 모델이 못 맞힘 · 저방문 n=91로 표본이 작음")

    with k2:
        tbl = tt[["미방문", "저방문", "고방문", "전체", "도달가능 전체", "NDCG@10", "저·미방문 노출비율", "Coverage", "ARP(평균 Train 방문수)"]]
        st.markdown("**Recall@10와 노출 지표 (Test, 전체 15개 방법)**")
        st.dataframe(tbl.style.format("{:.3f}").format("{:.1f}", subset=["ARP(평균 Train 방문수)"]))
        st.caption("미방문 열은 n=2라 참고용입니다. 도달가능 전체는 정답이 후보 풀 안에 있는 행만 본 값입니다.")
        extra = pd.DataFrame({
            "미방문": ["+0.000 [+0.000,+0.000]", "+0.000 [+0.000,+0.000]", "+0.000 [+0.000,+0.000]"],
            "저방문": ["+0.011 [+0.000,+0.033]", "+0.000 [-0.055,+0.055]", "+0.099 [+0.033,+0.176] ★"],
            "고방문": ["-0.014 [-0.028,+0.000]", "-0.046 [-0.093,+0.000]", "-0.051 [-0.111,+0.009]"],
            "전체": ["-0.005 [-0.016,+0.005]", "-0.027 [-0.056,+0.003]", "-0.005 [-0.043,+0.035]"],
        }, index=["4단계 최종 − 3단계 결합", "4단계 최종 − SASRec", "4단계 최종 − 지역+인기"])
        st.markdown("**차이와 95% 구간 (paired bootstrap, ★ = 구간이 0을 포함하지 않음)**")
        st.dataframe(pd.concat([pr, extra]))
        st.caption("저·미방문 노출비율의 4단계 − 3단계 차이는 +0.049 [+0.041, +0.057] ★ (보장 규칙이 만든 값). 위 표 앞 6행은 3단계 결합 기준, 뒤 3행은 4단계 최종 기준입니다.")
        rk = pd.DataFrame({
            "미방문": ["+0.000 [+0.000,+0.000]"] * 4,
            "저방문": ["+0.044 [+0.011,+0.088] ★", "-0.011 [-0.066,+0.044]", "+0.099 [+0.044,+0.165] ★", "+0.012 [-0.014,+0.042]"],
            "고방문": ["-0.005 [-0.051,+0.037]", "-0.032 [-0.074,+0.014]", "+0.032 [-0.009,+0.074]", "-0.026 [-0.060,+0.005]"],
            "전체": ["+0.008 [-0.019,+0.035]", "-0.021 [-0.051,+0.008]", "+0.043 [+0.013,+0.072] ★", "-0.012 [-0.033,+0.008]"],
        }, index=["Recall@5", "Recall@10", "Recall@20", "NDCG@10"])
        st.markdown("**평가 지표를 바꿔 보면 (3단계 결합 − SASRec)**")
        st.dataframe(rk)

    with k3:
        g = pd.read_csv(DATA / "rerank_val_grid.csv", encoding="utf-8-sig", index_col=0)
        g["규칙 통과"] = [("기준" if n.startswith("기준") else ("✅" if (r["Δ전체"] >= -0.01 and r["Δ고방문"] >= -0.03) else "❌")) for n, r in g.iterrows()]
        st.markdown("**재랭킹 설정 선택 (Val, 사전 규칙)** — 전체 Recall 손실 ≤ 0.01, 고방문 손실 ≤ 0.03을 만족하는 설정 중 저·미방문 노출이 가장 높은 것 (0.005 이내는 동점 → 더 약한 설정). 선택: **보장 m=2**")
        show = g[["저방문", "고방문", "전체", "Δ전체", "Δ고방문", "NDCG@10", "저·미방문 노출", "ARP", "Coverage", "Gini(노출)", "규칙 통과"]]
        st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if c != "규칙 통과"} | {"ARP": "{:.1f}"})
                     .apply(lambda r: ["background-color:#dff5df"] * len(r) if r.name == "보장 m=2" else [""] * len(r), axis=1))
        st.caption("보너스는 γ가 커질수록 노출이 오르지만 전체 Recall이 급락해 사전 규칙을 만족한 설정이 없었고, MMR은 노출이 거의 변하지 않아(0.378~0.383) 효과가 입증되지 않았습니다. "
                   "선택과 평가가 같은 Val이라 낙관적입니다.")

    with k4:
        figs = DATA / "figs"
        st.image(str(figs / "pipeline_final.png"), caption="최종 파이프라인")
        st.image(str(figs / "rerank_trade_off.png"), caption="재랭킹 trade-off (Val): 가로 전체 Recall@10, 세로 저·미방문 노출비율. 보장은 Recall을 유지하며 노출을 올리고, 보너스는 Recall이 급락, MMR은 변화 없음.")
        st.image(str(figs / "test_tradeoff_group.png"), caption="Test: 정확도 대비 노출(왼쪽)과 그룹별 Recall(오른쪽)")
        st.image(str(figs / "test_paired_forest.png"), caption="Test: paired 차이의 95% 범위. 선이 0을 가로지르면 '차이가 있다고 말할 수 없음'")

with t5:
    render_study(DATA)
