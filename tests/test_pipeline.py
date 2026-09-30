from src.application.use_cases.process_document import ProcessDocumentUseCase
from src.application.use_cases.validators import ExtractionValidator
from src.domain.entities.ocr import OcrLine, OcrResult
from src.infrastructure.database.sqlite_document_repository import SQLiteDocumentRepository
from src.infrastructure.image_processing.opencv_image_processor import OpenCvImageProcessor
from src.infrastructure.llm.rule_based_llm_engine import RuleBasedLlmEngine
from src.infrastructure.ocr.plain_text_ocr_engine import PlainTextOcrEngine
from src.infrastructure.storage.local_file_storage import LocalFileStorage
from src.infrastructure.text_detection.opencv_text_detector import OpenCvTextDetector
from src.infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore


def test_process_document_pipeline(tmp_path):
    repository = SQLiteDocumentRepository(tmp_path / "documents.db")
    vector_store = InMemoryVectorStore()
    use_case = ProcessDocumentUseCase(
        storage=LocalFileStorage(tmp_path / "uploads"),
        image_processor=OpenCvImageProcessor(tmp_path / "processed"),
        text_detector=OpenCvTextDetector(tmp_path / "text_regions"),
        ocr_engine=PlainTextOcrEngine(),
        llm_engine=RuleBasedLlmEngine(),
        repository=repository,
        vector_store=vector_store,
        validator=ExtractionValidator(),
        allowed_extensions={".txt"},
    )

    content = b"MINI MART AN PHU\nNgay ban: 12/08/2025\nTong cong: 92000 VND"
    result = use_case.execute("receipt.txt", "text/plain", content)

    assert result.status == "processed"
    assert result.image_quality["recommendation"] == "text_input_skipped"
    assert result.text_detection["region_count"] == 0
    assert result.extraction["document_type"] == "receipt"
    assert result.extraction["fields"]["total_amount"]["value"] == 92000
    assert repository.get_processing_result(result.document_id) is not None
    assert vector_store.search(result.document_id, "Tong tien") != []


def test_rule_based_answer_returns_direct_store_name():
    engine = RuleBasedLlmEngine()
    context = [
        "\n".join(
            [
                "THỨC COFFEE",
                "22 Quang Trung, P10, Gò Vấp",
                "Tiền Thanh Toán : 35 000",
            ]
        )
    ]

    answer = engine.answer("what's name coffee?", context)

    assert answer == "The shop name is THỨC COFFEE."


def test_rule_based_answer_returns_table_number_and_items():
    engine = RuleBasedLlmEngine()
    context = [
        "\n".join(
            [
                "THỨC COFFEE",
                "22 Quang Trung, P10, Gò Vấp",
                "PHIẾU TẠM TÍNH",
                "Số SỐ - 23 Số khách : 0",
                "TT Tên món SL Ð.Giá Ti Tier",
                "COFFEE WITH MILK -",
                "ICE",
                "1,00 35000 35 000",
                "Tiền Thanh Toán : 35 000",
            ]
        )
    ]

    assert engine.answer("bàn số mấy ?", context) == "The table number is 23."
    assert engine.answer("khách đã dùng món gì?", context) == "The customer ordered: COFFEE WITH MILK ICE."


def test_rule_based_extraction_detects_generic_document():
    engine = RuleBasedLlmEngine()
    result = engine.extract(
        OcrResult(
            lines=[
                OcrLine(text="Downloaded Report", confidence=0.95),
                OcrLine(text="This is a report downloaded in .docx format.", confidence=0.90),
            ]
        )
    )

    assert result.document_type == "document"
    assert result.fields["title"].value == "Downloaded Report"
