from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.presentation.api.dependencies import get_container
from src.shared.exceptions import AppError


def main() -> None:
    st.set_page_config(
        page_title="Vietnamese Document Intelligence",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_style()
    _init_state()

    st.title("Vietnamese Document Intelligence")

    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Document",
            type=["txt", "png", "jpg", "jpeg", "pdf"],
            accept_multiple_files=False,
        )
        run_clicked = st.button("Process", type="primary", use_container_width=True, disabled=uploaded_file is None)

        if uploaded_file is not None:
            st.caption(f"{uploaded_file.name} · {_format_bytes(uploaded_file.size)}")

    if run_clicked and uploaded_file is not None:
        _process_upload(uploaded_file)

    result = st.session_state.get("process_result")
    if result is None:
        _render_empty_state()
        return

    _render_result(result)


def _init_state() -> None:
    st.session_state.setdefault("process_result", None)
    st.session_state.setdefault("document_id", None)
    st.session_state.setdefault("chat_history", [])


def _process_upload(uploaded_file) -> None:
    container = get_container()
    use_case = container["process_document"]

    with st.spinner("Processing document"):
        try:
            result = use_case.execute(
                file_name=uploaded_file.name,
                content_type=uploaded_file.type or "application/octet-stream",
                content=uploaded_file.getvalue(),
            )
        except AppError as exc:
            st.error(str(exc))
            return

    result_dict = _model_to_dict(result)
    st.session_state["process_result"] = result_dict
    st.session_state["document_id"] = result.document_id
    st.session_state["chat_history"] = []


def _render_empty_state() -> None:
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader("Pipeline")
        st.markdown(
            """
            `Upload` -> `Quality Check` -> `Preprocessing` -> `Text Detection` -> `OCR` -> `LLM Extraction` -> `RAG`
            """
        )
    with right:
        st.subheader("Sample")
        st.code("samples/images/receipt_sample.png", language="text")


def _render_result(result: Dict[str, Any]) -> None:
    metrics = st.columns(5)
    metrics[0].metric("Status", result.get("status", "-"))
    metrics[1].metric("OCR confidence", f"{result.get('average_ocr_confidence', 0):.2f}")
    metrics[2].metric("Quality", f"{result.get('image_quality', {}).get('quality_score', 0):.2f}")
    metrics[3].metric("Text regions", result.get("text_detection", {}).get("region_count", 0))
    metrics[4].metric("Validation", result.get("validation", {}).get("status", "-"))

    tab_summary, tab_ocr, tab_extraction, tab_chat, tab_json = st.tabs(
        ["Summary", "OCR", "Extraction", "Ask", "JSON"]
    )

    with tab_summary:
        _render_summary(result)
    with tab_ocr:
        _render_ocr(result)
    with tab_extraction:
        _render_extraction(result)
    with tab_chat:
        _render_chat(result)
    with tab_json:
        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")


def _render_summary(result: Dict[str, Any]) -> None:
    left, right = st.columns(2, gap="large")
    image_quality = result.get("image_quality", {})
    text_detection = result.get("text_detection", {})

    with left:
        st.subheader("Image Quality")
        st.table(
            [
                {"metric": "quality_score", "value": image_quality.get("quality_score")},
                {"metric": "recommendation", "value": image_quality.get("recommendation")},
                {"metric": "issues", "value": ", ".join(image_quality.get("issues", [])) or "-"},
                {"metric": "was_processed", "value": image_quality.get("was_processed")},
            ]
        )

    with right:
        st.subheader("Text Detection")
        regions = text_detection.get("regions", [])
        st.table(
            [
                {
                    "region": index + 1,
                    "bbox": region.get("bbox"),
                    "confidence": region.get("confidence"),
                }
                for index, region in enumerate(regions[:8])
            ]
            or [{"region": "-", "bbox": "-", "confidence": "-"}]
        )


def _render_ocr(result: Dict[str, Any]) -> None:
    st.subheader("OCR Text")
    st.text_area("ocr_text", result.get("ocr_text", ""), height=260, label_visibility="collapsed")


def _render_extraction(result: Dict[str, Any]) -> None:
    extraction = result.get("extraction", {})
    fields = extraction.get("fields", {})
    st.subheader(f"Document Type: {extraction.get('document_type', 'unknown')}")

    rows: List[Dict[str, Any]] = []
    for name, field in fields.items():
        rows.append(
            {
                "field": name,
                "value": field.get("value"),
                "confidence": field.get("confidence"),
                "needs_review": field.get("needs_review"),
                "source_text": field.get("source_text"),
            }
        )

    if rows:
        st.markdown(_markdown_table(rows))
    else:
        st.info("No extracted fields.")

    validation = result.get("validation", {})
    if validation.get("needs_review"):
        st.warning("Needs review: " + ", ".join(validation["needs_review"]))
    if validation.get("errors"):
        st.error("Errors: " + ", ".join(validation["errors"]))


def _render_chat(result: Dict[str, Any]) -> None:
    document_id = result.get("document_id")
    if not document_id:
        st.info("No document selected.")
        return

    for item in st.session_state.get("chat_history", []):
        st.markdown(f"**You:** {_escape_markdown(str(item['question']))}")
        st.markdown(f"**Assistant:** {_escape_markdown(str(item['answer']))}")
        if item.get("sources"):
            with st.expander("Sources"):
                for source in item["sources"]:
                    st.code(source, language="text")

    question = st.text_input("Ask about this document", key=f"question_{document_id}")
    if not question:
        return
    if not st.button("Ask", type="primary"):
        return

    container = get_container()
    use_case = container["ask_document"]
    try:
        answer = use_case.execute(document_id=document_id, question=question)
    except AppError as exc:
        st.error(str(exc))
        return

    answer_dict = _model_to_dict(answer)
    st.session_state["chat_history"].append(answer_dict)
    st.rerun()


def _model_to_dict(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _markdown_table(rows: List[Dict[str, Any]]) -> str:
    headers = ["field", "value", "confidence", "needs_review", "source_text"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [_escape_table_value(row.get(header)) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _escape_table_value(value: Any) -> str:
    if value is None:
        return "-"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _escape_markdown(value: str) -> str:
    return value.replace("\\", "\\\\").replace("*", "\\*").replace("_", "\\_")


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            padding: 0.75rem;
            border-radius: 8px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
