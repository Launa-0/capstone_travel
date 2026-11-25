"""'왜 이렇게 만들었나' 탭: EDA·3단계·4단계 결과를 데모의 설계 근거로 풀어 쓴 요약."""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


def _img(data_dir: Path, name: str, caption: str):
    p = Path(data_dir) / "figs" / name
    if p.exists():
        try:
            st.image(str(p), caption=caption, width="stretch")
        except Exception:
            st.image(str(p), caption=caption, use_container_width=True)


def _chart(ch):
    try:
        st.altair_chart(ch, width="stretch")
    except Exception:
        st.altair_chart(ch, use_container_width=True)


def _bias_charts(data_dir):
    v = pd.read_csv(Path(data_dir) / "poi.csv")["train_visits"].to_numpy()
    v = np.sort(v[v > 0])[::-1]
    n = len(v)
    rank = pd.DataFrame({"순위": np.arange(1, n + 1), "방문수": v})
    asc = v[::-1]
    c = np.r_[0, np.cumsum(asc) / asc.sum()]
    lor = pd.DataFrame({"장소 누적 비율": np.arange(n + 1) / n, "방문 누적 비율": c})
    diag = pd.DataFrame({"장소 누적 비율": [0, 1], "방문 누적 비율": [0, 1]})
    grp = pd.DataFrame({"그룹": ["미방문", "저방문", "고방문"], "비중(%)": [13.3, 26.5, 60.2]})
    a, b, d = st.columns(3)
    with a:
        _chart(alt.Chart(rank, title="순위-방문수 (로그-로그)", height=260).mark_line().encode(
            x=alt.X("순위:Q", scale=alt.Scale(type="log")), y=alt.Y("방문수:Q", scale=alt.Scale(type="log"))))
    with b:
        l1 = alt.Chart(lor, title="로렌츠 곡선 (지니 0.622)", height=260).mark_line().encode(x="장소 누적 비율:Q", y="방문 누적 비율:Q")
        l2 = alt.Chart(diag).mark_line(strokeDash=[5, 4], color="#f58518").encode(x="장소 누적 비율:Q", y="방문 누적 비율:Q")
        _chart(l1 + l2)
    with d:
        _chart(alt.Chart(grp, title="검증 정답의 방문 그룹 비중", height=260).mark_bar().encode(
            x=alt.X("그룹:N", sort=["미방문", "저방문", "고방문"], axis=alt.Axis(labelAngle=0)), y="비중(%):Q",
            tooltip=["그룹", "비중(%)"]))
    st.caption("방문 상위 10% 장소가 방문의 55.3%를 차지하고, 방문이 1회뿐인 장소가 54.4%입니다(학습 방문이 있는 3,136곳 기준).")


def _dist_chart():
    df = pd.DataFrame({
        "그룹": ["미방문", "저방문", "고방문"] * 2,
        "방식": ["거리 기준 10곳"] * 3 + ["SASRec"] * 3,
        "Recall@10": [0.214, 0.281, 0.155, 0.000, 0.129, 0.547],
    })
    _chart(alt.Chart(df, title="그룹별 Recall@10: 직전 방문지에서 가까운 10곳 vs SASRec (Val)", height=280).mark_bar().encode(
        x=alt.X("그룹:N", sort=["미방문", "저방문", "고방문"], axis=alt.Axis(labelAngle=0), title=None),
        xOffset="방식:N", y="Recall@10:Q", color="방식:N", tooltip=["그룹", "방식", "Recall@10"]))


def render_study(data_dir):
    st.caption("위 추천 화면의 각 설정이 어떤 데이터 관찰과 실험에서 나왔는지 순서대로 정리했습니다.")

    # ---------------------------------------------------------------- 1. 문제
    st.subheader("1. 문제: 인기 장소만 추천되는 구조")
    c1, c2, c3 = st.columns(3)
    c1.metric("방문 상위 10% 장소의 방문 비중", "55.3%")
    c2.metric("방문이 1회뿐인 장소", "54.4%")
    c3.metric("지니계수", "0.622")
    st.markdown(
        "후보 3,389곳 중 약 78%가 방문 5회 이하(저방문)인데, 전체 방문에서 차지하는 몫은 36%에 그칩니다. "
        "방문 이력만 학습하는 SASRec은 이 편향을 그대로 따라가서, 검증 데이터 기준 저방문 정답은 "
        "Recall@10 0.129, 한 번도 방문된 적 없는 장소(미방문)는 0.000이었습니다."
    )
    _bias_charts(data_dir)

    # ---------------------------------------------------------------- 2. 신호
    st.subheader("2. 이력 말고 쓸 수 있는 신호")
    st.markdown(
        "- **거리**: 직전 방문지에서 가까운 10곳만 골라도 저방문 0.281, 미방문 0.214로 SASRec보다 높았습니다. "
        "정답은 직전 방문지에서 중앙값 약 4.8km, 90%가 41km 안에 있었습니다.\n"
        "- **장소 설명 텍스트**: 방문 이유·활동 코드는 거의 모든 장소에 있지만, 마스터의 종별 분류는 94~96%가 비어 "
        "있고 저방문 장소의 서술은 짧습니다(중앙값 7자). 그래서 분류는 버리고 이름·지역·활동 코드를 문장으로 묶어 "
        "ko-sroberta로 임베딩했습니다.\n"
        "- **지역**: 정답이 직전 방문지와 같은 시도인 비율이 89.5%지만, SASRec이 이미 이를 반영하고 있어 "
        "별도 항으로는 쓰지 않았습니다.\n"
        "- **개인 스타일 설문**: 일관된 신호가 약해 모델에 넣지 않았습니다(구현·검증하지 않은 것이며 후속 과제입니다)."
    )
    _dist_chart()

    # ---------------------------------------------------------------- 3. 점수 결합
    st.subheader("3. 점수 결합: 시퀀스 1 + 텍스트 1 + 거리 0.25")
    st.markdown(
        "세 점수를 후보마다 z-점수로 맞춘 뒤 가중합합니다. 방문 이력이 없는 장소는 시퀀스 항이 없으므로 "
        "그 항을 빼고 가중치를 다시 나눕니다(그래서 미방문도 점수를 받습니다). "
        "가중치는 검증(Val)에서 사전에 정한 규칙으로 골랐고, 테스트는 모든 설정을 확정한 뒤 한 번만 열었습니다."
    )
    st.markdown(
        "| 시도 | 결과 | 결정 |\n|---|---|---|\n"
        "| 텍스트 점수 (ko-sroberta) | Val에서 저·고방문 모두 기여(★), Test에서는 확인 안 됨 | 채택 |\n"
        "| 거리 점수 | EDA 신호는 강하나 결합에서 거리 제거 차이는 0을 포함 | 유지하되 입증 안 됨으로 기록 |\n"
        "| 협업 필터링(CF) | 고방문 −0.054 ★, 전체 −0.027 | 기각 |\n"
        "| e5-base, mpnet-multi 임베딩 | 모든 그룹에서 ko-sroberta보다 낮음 | 기각 |\n"
        "| 마스터 종별 분류 | 94~96% 결측 | 기각 |\n"
        "| TourAPI 보강 | 표본 매칭률 약 10% | 기각 |\n"
        "| rank 정규화 | 고방문 Recall 0.453으로 손실 큼 | 기각 |"
    )
    st.info(
        "검증에서는 저방문 Recall이 0.129 → 0.194로 올랐지만, 테스트에서는 이 이득이 재현되지 않았습니다"
        "(전체 SASRec 0.416, 결합 0.394, 차이는 통계적으로 구분되지 않음). "
        "그래서 결합의 가치는 '정확도 향상'이 아니라 저·미방문 노출과 지역+인기 기준선 대비 저방문 우위로 서술합니다."
    )

    # ---------------------------------------------------------------- 4. 재랭킹
    st.subheader("4. 재랭킹: 저·미방문 최소 보장 (m=2)")
    st.markdown(
        "관련성 상위 50개 안에서 Top-10에 저·미방문이 2개 미만이면, 10개 중 가장 낮은 고방문 장소를 빼고 "
        "후보 안의 저·미방문을 관련성 순으로 채웁니다(교체된 장소는 6~10번째 자리에 들어갑니다). "
        "정답 정보는 쓰지 않고, 인기도는 학습 방문수로 그룹을 나눌 때만 씁니다."
    )
    st.markdown(
        "| 시도 | 결과 | 결정 |\n|---|---|---|\n"
        "| 인기도 역가중치 보너스 | γ=0.1에서도 전체 −0.015~−0.027, 사전 규칙 통과 설정 0개 | 기각(비교용만) |\n"
        "| MMR 다양성 | 노출 0.378~0.383(적용 전 0.377), Recall 변화 ±0.004 | 기각 |\n"
        "| 최소 보장 m=2 | 노출 0.377 → 0.425(Val), Recall 손실 작음 | **채택** |\n"
        "| 표시 개수 Top-5 | 보장이 10개 목록 기준이라 5개에서는 효과가 사라짐 | Top-10으로 변경 |"
    )
    _img(data_dir, "rerank_trade_off.png", "재랭킹 설정별 Recall과 저·미방문 노출의 맞교환")

    # ---------------------------------------------------------------- 5. 한계
    st.subheader("5. 한계와 주의")
    st.markdown(
        "- **후보 풀**: 학습·검증에 나온 3,389곳만 추천할 수 있습니다. 테스트 정답의 17.2%(64행)는 이 풀 밖이라 "
        "어떤 모델도 맞힐 수 없고, 미방문 평가는 테스트에 2건뿐이라 불가능합니다.\n"
        "- **연도별 장소 분리**: 2022와 2023에서 같은 장소가 다른 ID로 나뉜 정황(이름 같은 쌍 84개)이 있으나 "
        "병합하지 않았습니다.\n"
        "- **좌표 오류**: 방문 좌표가 30km 이상 흩어진 장소가 약 6%입니다. 이런 장소는 추천 표에 ⚠로 표시하며 "
        "거리·텍스트 점수가 영향을 받을 수 있습니다.\n"
        "- **주장하지 않는 것**: 결합 모델이 SASRec보다 정확하다, 텍스트·거리가 Recall을 높인다, 미방문이 개선됐다."
    )
