"""연구 보고서 화면: 노션 보고서 3편(EDA · 3단계 · 4단계)을 목차별로 보여 준다.

보고서 원문은 demo_data/reports/*.md (노션 내보내기 그대로), 그림은 demo_data/figs/*.png.
"""
import re
from pathlib import Path

import streamlit as st

REPORTS = {
    "① 데이터 이해 · EDA": ("eda.md", "Part I(데이터 개요·ERD·컬럼 해석)와 Part II(EDA 분석 결과). 1차 분석 기록이며, 맨 위 '확정 사항'이 최종 방식입니다."),
    "② 3단계: 텍스트 점수와 점수 결합": ("step3.md", "장소 문서·임베딩, 텍스트·거리 점수, 가중치 선택, ablation, 채택·기각 정리, Test에서 확인된 것."),
    "③ 4단계: 재랭킹과 최종 Test": ("step4.md", "저·미방문 보장 재랭킹, 기각한 방법(보너스·MMR), Val 선택표, 최종 Test 결과와 판정."),
}
IMG_ALIAS = {"image": "pipeline_final"}          # 노션의 '!image.png' = 파이프라인 그림


def _sections(text: str):
    """H1/H2 기준으로 (라벨, 본문) 목록을 만든다. 머리말은 '개요'로 둔다."""
    lines = text.splitlines()
    out, cur_label, cur, h1 = [], "개요 (머리말)", [], ""
    for ln in lines:
        m = re.match(r"^(#{1,2})\s+(.*)", ln)
        if m:
            if "".join(cur).strip():
                out.append((cur_label, "\n".join(cur)))
            lvl, title = len(m.group(1)), m.group(2).strip()
            if lvl == 1:
                h1 = re.split(r"[:·.]", title)[0].strip() if title.startswith("Part") else title
                cur_label = title
            else:
                cur_label = f"{h1} · {title}" if h1.startswith("Part") else title
            cur = [ln]
        else:
            cur.append(ln)
    if "".join(cur).strip():
        out.append((cur_label, "\n".join(cur)))
    # 제목 한 줄뿐인 H1 구간은 뺀다
    return [(a, b) for a, b in out if len([x for x in b.splitlines() if x.strip()]) > 1]


def _md(block: str):
    block = block.replace("$", r"\$")
    # **”문장”** 처럼 따옴표가 굵게 표시 기호에 붙어 있으면 마크다운이 굵게 처리하지 못해서, 따옴표를 바깥으로 뺀다
    block = re.sub(r"\*\*([“”\"'‘’])(.+?)([“”\"'‘’])\*\*", r"\1**\2**\3", block)
    st.markdown(block)


def _render_plain(block: str, figs: Path):
    """이미지 줄(!name.svg)은 st.image로, 나머지는 마크다운으로."""
    buf = []
    for ln in block.splitlines():
        m = re.match(r"^!(\S+?)\.(svg|png|jpg)\s*$", ln.strip())
        if m:
            if buf:
                _md("\n".join(buf))
                buf = []
            name = IMG_ALIAS.get(m.group(1), m.group(1))
            p = figs / f"{name}.png"
            if p.exists():
                st.image(str(p))
            else:
                st.caption(f"(그림 파일 없음: {name})")
        else:
            buf.append(ln)
    if buf:
        _md("\n".join(buf))


def render_body(body: str, figs: Path):
    """<aside> ... </aside> 는 테두리 상자로."""
    pos = 0
    for m in re.finditer(r"<aside>(.*?)</aside>", body, flags=re.S):
        pre = body[pos:m.start()]
        if pre.strip():
            _render_plain(pre, figs)
        with st.container(border=True):
            _render_plain(m.group(1).strip(), figs)
        pos = m.end()
    rest = body[pos:]
    if rest.strip():
        _render_plain(rest, figs)


def render_reports(data_dir: Path):
    figs = data_dir / "figs"
    st.title("📚 연구 보고서")
    st.caption("팀 노션 보고서를 그대로 옮긴 화면입니다. 수치·판정은 원문 기준이며, 이 앱이 새로 계산한 값이 아닙니다.")
    name = st.sidebar.radio("보고서", list(REPORTS))
    fname, desc = REPORTS[name]
    st.sidebar.caption(desc)
    text = (data_dir / "reports" / fname).read_text(encoding="utf-8")
    secs = _sections(text)
    labels = ["전체 보기"] + [a for a, _ in secs]
    pick = st.sidebar.selectbox("목차", labels, index=1, help="'전체 보기'는 긴 문서라 느릴 수 있습니다.")
    st.subheader(name)
    q = st.sidebar.text_input("보고서에서 검색", placeholder="예: 지역+인기, MMR, 미방문")
    if q:
        hits = [(a, b) for a, b in secs if q.lower() in b.lower()]
        st.markdown(f"**‘{q}’가 들어 있는 구간 {len(hits)}개**")
        for a, b in hits:
            with st.expander(a):
                render_body(b, figs)
        return
    shown = secs if pick == "전체 보기" else [(a, b) for a, b in secs if a == pick]
    for a, b in shown:
        render_body(b, figs)
        if a.startswith("부록") and fname == "eda.md":
            with st.expander("그림 15개 전체 보기"):
                pngs = sorted(p for p in figs.glob("fig*.png"))
                for i in range(0, len(pngs), 2):
                    cols = st.columns(2)
                    for c, p in zip(cols, pngs[i:i + 2]):
                        c.image(str(p), caption=p.stem)
