from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# 나중에 학습한 모델 경로로 바꿀 수 있습니다.
DEFAULT_MODEL = "yolo26n.pt"
RESULT_DIR = Path("result")
HISTORY_CSV = RESULT_DIR / "counts.csv"

st.set_page_config(
    page_title="나의 소비패턴",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 나의 소비패턴")
st.caption(
    "이미지를 업로드하거나 카메라로 촬영하면 객체를 탐지하고 상품별 수량을 기록합니다."
)


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


def parse_history_dates(values: pd.Series) -> pd.Series:
    """시간대가 없는 기존 기록은 한국 시각으로 해석합니다."""
    parsed_values = []

    for value in values:
        parsed = pd.to_datetime(value, errors="coerce")

        if pd.isna(parsed):
            parsed_values.append(pd.NaT)
        elif parsed.tzinfo is None:
            parsed_values.append(parsed.tz_localize("Asia/Seoul"))
        else:
            parsed_values.append(parsed)

    return pd.Series(
        pd.to_datetime(parsed_values, errors="coerce", utc=True),
        index=values.index,
    )


def sort_history(history: pd.DataFrame) -> pd.DataFrame:
    """날짜를 실제 시각으로 해석해 최근 기록부터 정렬합니다."""
    if history.empty:
        return history.reset_index(drop=False)

    sorted_history = history.copy()
    sorted_history["_정렬용 날짜"] = parse_history_dates(sorted_history["날짜"])

    sorted_history = sorted_history.sort_values(
        by="_정렬용 날짜",
        ascending=False,
        na_position="last",
        kind="stable",
    )

    return (
        sorted_history
        .drop(columns="_정렬용 날짜")
        .reset_index(drop=True)
    )


def history_for_display(history: pd.DataFrame) -> pd.DataFrame:
    """한국 시간으로 변환하고 초까지만 화면과 다운로드에 표시합니다."""
    display_history = history.copy()

    parsed_dates = parse_history_dates(display_history["날짜"])

    display_history["날짜"] = (
        parsed_dates.dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
        .fillna(display_history["날짜"].astype(str))
    )

    return display_history


def read_history() -> pd.DataFrame:
    """CSV에서 누적 기록을 읽고 최신순으로 정렬합니다."""
    columns = ["날짜", "상품", "수량"]

    if not HISTORY_CSV.exists():
        return pd.DataFrame(columns=columns)

    try:
        history = pd.read_csv(HISTORY_CSV, encoding="utf-8-sig")

        if not set(columns).issubset(history.columns):
            return pd.DataFrame(columns=columns)

        return sort_history(history[columns])

    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=columns)


def append_history(records: list[dict]) -> None:
    """새 기록을 CSV에 추가하고 최신순으로 저장합니다."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    history = read_history()
    additions = pd.DataFrame(records, columns=["날짜", "상품", "수량"])
    combined = pd.concat([history, additions], ignore_index=True)
    combined = sort_history(combined)

    combined.to_csv(
        HISTORY_CSV,
        index=False,
        encoding="utf-8-sig",
    )


with st.sidebar:
    st.header("모델 설정")

    model_path = st.text_input(
        "모델 파일 또는 모델 이름",
        value=DEFAULT_MODEL,
    )

    confidence = st.slider(
        "감지 신뢰도",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )

    st.caption(
        "기본 모델은 일반 객체 클래스를 인식합니다. "
        "상품별 인식 모델은 나중에 경로를 바꾸면 됩니다."
    )


st.subheader("1. 이미지 입력")

input_mode = st.radio(
    "입력 방식",
    ["사진 업로드", "카메라 촬영"],
    horizontal=True,
)

if input_mode == "사진 업로드":
    image_file = st.file_uploader(
        "상품 사진을 선택하세요",
        type=["jpg", "jpeg", "png", "webp"],
    )
else:
    image_file = st.camera_input("상품 사진을 촬영하세요")


if image_file is not None:
    pil_image = Image.open(image_file).convert("RGB")

    st.image(
        pil_image,
        caption="분석할 이미지",
        use_container_width=True,
    )

    if st.button("상품 분석", type="primary"):
        try:
            import cv2
            import numpy as np

            model = load_model(model_path.strip())

            # Ultralytics 입력 형식에 맞춰 RGB 이미지를 BGR로 변환합니다.
            image_array = cv2.cvtColor(
                np.array(pil_image),
                cv2.COLOR_RGB2BGR,
            )

            with st.spinner("이미지를 분석하고 있습니다..."):
                prediction = model.predict(
                    image_array,
                    conf=confidence,
                    verbose=False,
                )[0]

            # 탐지 결과 이미지를 표시합니다.
            annotated_bgr = prediction.plot()
            annotated_rgb = cv2.cvtColor(
                annotated_bgr,
                cv2.COLOR_BGR2RGB,
            )

            st.subheader("2. 탐지 결과")
            st.image(
                annotated_rgb,
                caption="객체 탐지 결과",
                use_container_width=True,
            )

            # 감지된 클래스별 수량을 계산합니다.
            counts: dict[str, int] = {}

            if prediction.boxes is not None and prediction.boxes.cls is not None:
                class_ids = prediction.boxes.cls.int().cpu().tolist()

                for class_id in class_ids:
                    name = str(prediction.names[class_id])
                    counts[name] = counts.get(name, 0) + 1

            st.subheader("3. 이번 이미지의 집계")

            # 배포 서버 시간대와 관계없이 한국 시각으로 저장합니다.
            detected_at = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(
                timespec="microseconds"
            )

            if counts:
                current = pd.DataFrame(
                    [
                        {
                            "날짜": detected_at,
                            "상품": name,
                            "수량": count,
                        }
                        for name, count in sorted(counts.items())
                    ]
                )

                st.dataframe(
                    history_for_display(current),
                    use_container_width=True,
                    hide_index=True,
                )

                append_history(current.to_dict(orient="records"))
                st.success("집계를 저장했습니다.")

            else:
                st.info(
                    "감지된 객체가 없습니다. "
                    "신뢰도 설정이나 이미지 구도를 확인해 보세요."
                )

                no_detection = pd.DataFrame(
                    [
                        {
                            "날짜": detected_at,
                            "상품": "감지된 객체 없음",
                            "수량": 0,
                        }
                    ]
                )

                append_history(no_detection.to_dict(orient="records"))

        except Exception as error:
            st.error(f"분석 중 오류가 발생했습니다: {error}")


st.divider()
st.subheader("저장된 전체 기록")

history = read_history()

if history.empty:
    st.info("아직 저장된 기록이 없습니다. 사진을 분석하면 이곳에 누적됩니다.")
else:
    display_history = history_for_display(history)

    st.dataframe(
        display_history,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "전체 기록 CSV 다운로드",
        data=display_history.to_csv(index=False).encode("utf-8-sig"),
        file_name="counts.csv",
        mime="text/csv",
        key="download_all_history",
    )


if st.button("전체 기록 초기화", type="secondary"):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(columns=["날짜", "상품", "수량"]).to_csv(
        HISTORY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    st.success("저장된 전체 기록을 초기화했습니다.")
    st.rerun()