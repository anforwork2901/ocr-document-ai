import re
from datetime import datetime
from typing import Dict, List, Optional

from src.domain.entities.extraction import ExtractedField, ExtractionResult, FieldConfidence
from src.domain.entities.ocr import OcrResult
from src.domain.services.llm_engine import LlmEngine


class RuleBasedLlmEngine(LlmEngine):
    """Deterministic local implementation of the LLM port for MVP demos."""

    def extract(self, ocr_result: OcrResult) -> ExtractionResult:
        text = ocr_result.text
        fields: Dict[str, ExtractedField] = {}
        document_type = self._detect_document_type(text)

        if document_type != "receipt":
            title = self._first_non_empty_line(text)
            if title:
                fields["title"] = ExtractedField(
                    value=title,
                    source_text=title,
                    confidence=FieldConfidence.HIGH if ocr_result.average_confidence >= 0.85 else FieldConfidence.MEDIUM,
                    needs_review=ocr_result.average_confidence < 0.85,
                )
            return ExtractionResult(document_type=document_type, fields=fields)

        store_name = self._find_store_name_source(text)
        if store_name:
            fields["store_name"] = ExtractedField(
                value=store_name,
                source_text=store_name,
                confidence=FieldConfidence.MEDIUM,
                needs_review=ocr_result.average_confidence < 0.85,
            )

        date_source = self._find_date_source(text)
        if date_source:
            fields["purchase_date"] = ExtractedField(
                value=self._normalize_date(date_source),
                source_text=date_source,
                confidence=FieldConfidence.HIGH,
            )

        total_source = self._find_total_source(text)
        if total_source:
            fields["total_amount"] = ExtractedField(
                value=self._parse_amount(total_source),
                source_text=total_source,
                confidence=FieldConfidence.HIGH,
            )

        return ExtractionResult(document_type=document_type, fields=fields)

    def answer(self, question: str, context: List[str]) -> str:
        if not context:
            return "I could not find relevant information in this document."
        context_text = "\n".join(context)
        normalized_question = question.lower()

        if self._is_store_name_question(normalized_question):
            store_name = self._find_store_name_source(context_text)
            if store_name:
                return f"The shop name is {store_name}."

        if self._is_address_question(normalized_question):
            address = self._find_address_source(context_text)
            if address:
                return f"The address is {address}."

        if self._is_date_question(normalized_question):
            date_source = self._find_date_source(context_text)
            if date_source:
                return f"The purchase date is {self._normalize_date(date_source)}."

        if self._is_table_question(normalized_question):
            table_number = self._find_table_number(context_text)
            if table_number:
                return f"The table number is {table_number}."

        if self._is_items_question(normalized_question):
            items = self._find_ordered_items(context_text)
            if items:
                return "The customer ordered: " + ", ".join(items) + "."

        total = self._parse_amount(self._find_total_source(context_text) or "")
        if total and self._is_total_question(normalized_question):
            return f"The total amount in the document is {total:,} VND."
        return "I found the document, but I could not extract a direct answer from the retrieved context."

    def _first_non_empty_line(self, text: str) -> Optional[str]:
        for line in text.splitlines():
            cleaned = line.strip()
            if self._is_meaningful_text_line(cleaned):
                return re.split(r"\bngay\b|\bngày\b", cleaned, flags=re.IGNORECASE)[0].strip()
        return None

    def _detect_document_type(self, text: str) -> str:
        normalized = text.lower()
        receipt_keywords = [
            "hóa đơn",
            "hoa don",
            "phiếu tạm tính",
            "phieu tam tinh",
            "tổng tiền",
            "tong tien",
            "tổng cộng",
            "tong cong",
            "thanh toán",
            "thanh toan",
            "mặt hàng",
            "mat hang",
            "ngày bán",
            "ngay ban",
            "tiền khách",
            "tien khach",
            "vincommerce",
            "vinmart",
        ]
        return "receipt" if any(keyword in normalized for keyword in receipt_keywords) else "document"

    def _find_store_name_source(self, text: str) -> Optional[str]:
        lines = [line.strip(" `{}-_—") for line in text.splitlines() if line.strip()]
        known_brands = ["VinCommerce", "VinMart", "THỨC COFFEE", "MINIMART", "MINI MART"]
        for line in lines:
            for brand in known_brands:
                if brand.lower() in line.lower():
                    return brand
        for line in lines:
            normalized = line.lower()
            if any(keyword in normalized for keyword in ["hóa đơn", "hoa don", "ngày", "ngay", "mặt hàng", "mat hang"]):
                break
            if self._is_meaningful_text_line(line):
                return re.split(r"\bngay\b|\bngày\b", line, flags=re.IGNORECASE)[0].strip()
        return self._first_non_empty_line(text)

    def _is_meaningful_text_line(self, line: str) -> bool:
        if not line or not re.search(r"[A-Za-zÀ-ỹ]", line):
            return False
        letters = sum(char.isalpha() for char in line)
        visible = sum(not char.isspace() for char in line)
        return visible > 0 and letters / visible >= 0.45

    def _find_address_source(self, text: str) -> Optional[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            normalized = line.lower()
            if any(keyword in normalized for keyword in ["địa chỉ", "dia chi", "address"]):
                return line
        if len(lines) > 1:
            return lines[1]
        return None

    def _find_date_source(self, text: str) -> Optional[str]:
        match = re.search(r"\b\d{1,2}[./-][O0]?\d{1,2}[./-]\d{2,4}\b", text, re.IGNORECASE)
        if match:
            return match.group(0)
        vietnamese_date = re.search(
            r"ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}",
            text,
            re.IGNORECASE,
        )
        return vietnamese_date.group(0) if vietnamese_date else None

    def _normalize_date(self, source: str) -> str:
        cleaned = source.replace("O", "0").replace("o", "0")
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y", "%d.%m.%y"):
            try:
                return datetime.strptime(cleaned, fmt).date().isoformat()
            except ValueError:
                continue
        vietnamese_date = re.search(
            r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
            cleaned,
            re.IGNORECASE,
        )
        if vietnamese_date:
            day, month, year = vietnamese_date.groups()
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        return cleaned

    def _find_total_source(self, text: str) -> Optional[str]:
        candidates = []
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            normalized = line.lower()
            has_total_keyword = any(
                keyword in normalized
                for keyword in [
                    "tong",
                    "tổng",
                    "total",
                    "thanh toán",
                    "thanh toan",
                    "phải trả",
                    "phai tra",
                    "khach tra",
                    "khách trả",
                ]
            )
            has_amount = bool(self._find_amount_candidates(line))
            if has_total_keyword and has_amount:
                candidates.append(line)
            elif has_total_keyword:
                nearby = " ".join(lines[index : index + 3])
                if self._find_amount_candidates(nearby):
                    candidates.append(nearby)
        return candidates[-1] if candidates else None

    def _find_table_number(self, text: str) -> Optional[str]:
        compact_text = " ".join(line.strip() for line in text.splitlines() if line.strip())
        patterns = [
            r"\bs[ốo]\s*(?:bàn|ban|s[ốo])?\s*[-:]?\s*(\d{1,4})\b",
            r"\bbàn\s*(?:s[ốo])?\s*[-:]?\s*(\d{1,4})\b",
            r"\bban\s*(?:so)?\s*[-:]?\s*(\d{1,4})\b",
            r"\btable\s*(?:no\.?|number)?\s*[-:]?\s*(\d{1,4})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, compact_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _find_ordered_items(self, text: str) -> List[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        items: List[str] = []
        in_items_section = False
        pending_item: Optional[str] = None

        for line in lines:
            normalized = line.lower()
            if any(keyword in normalized for keyword in ["tên món", "ten mon", "mặt hàng", "mat hang", "item", "description"]):
                in_items_section = True
                continue
            if in_items_section and any(
                keyword in normalized
                for keyword in ["tiền thanh toán", "tien thanh toan", "tổng", "tong", "total", "khách đưa", "khach dua"]
            ):
                break
            if not in_items_section:
                continue

            candidate = self._clean_item_line(line)
            if not candidate:
                continue

            if self._looks_like_item_continuation(candidate) and pending_item:
                pending_item = f"{pending_item} {candidate}".strip()
                continue

            if pending_item:
                items.append(pending_item)
            pending_item = candidate

        if pending_item:
            items.append(pending_item)

        deduped = []
        for item in items:
            if item not in deduped:
                deduped.append(item)
        return deduped[:5]

    def _clean_item_line(self, line: str) -> Optional[str]:
        candidate = re.sub(r"\b\d+[,.]\d+\b", " ", line)
        candidate = re.sub(r"\b\d[\d\s.,]{2,}\b", " ", candidate)
        candidate = re.sub(r"[^0-9A-Za-zÀ-ỹ&/().% -]", " ", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip(" -")
        if len(candidate) < 2:
            return None
        if candidate.isdigit():
            return None
        if not re.search(r"[A-Za-zÀ-ỹ]", candidate):
            return None
        noisy_words = {"tt", "sl", "gia", "giá", "tier", "ti", "d.gia", "đ.giá"}
        words = {word.lower().strip(".") for word in candidate.split()}
        if words and words.issubset(noisy_words):
            return None
        return candidate

    def _looks_like_item_continuation(self, candidate: str) -> bool:
        return len(candidate.split()) <= 2 and not re.search(r"\d", candidate)

    def _parse_amount(self, source: str) -> Optional[int]:
        cleaned = source.replace("O", "0").replace("o", "0")
        numbers = self._find_amount_candidates(cleaned)
        if not numbers:
            return None
        raw = re.sub(r"[^\d]", "", numbers[-1])
        if not raw:
            return None
        return int(raw)

    def _find_amount_candidates(self, source: str) -> List[str]:
        candidates = re.findall(r"\b\d{1,3}(?:[.,\s]\d{3})+\b", source)
        candidates.extend(re.findall(r"\b\d{4,}\b", source))
        return [candidate for candidate in candidates if int(re.sub(r"[^\d]", "", candidate) or "0") > 0]

    def _is_store_name_question(self, question: str) -> bool:
        store_keywords = ["store", "shop", "merchant", "restaurant", "coffee", "cafe", "quán", "quan", "cửa hàng"]
        name_keywords = ["name", "tên", "ten"]
        return any(keyword in question for keyword in store_keywords) and any(
            keyword in question for keyword in name_keywords
        )

    def _is_address_question(self, question: str) -> bool:
        return any(keyword in question for keyword in ["address", "địa chỉ", "dia chi", "ở đâu", "o dau"])

    def _is_date_question(self, question: str) -> bool:
        return any(keyword in question for keyword in ["date", "ngày", "ngay", "when", "khi nào", "khi nao"])

    def _is_total_question(self, question: str) -> bool:
        return any(keyword in question for keyword in ["total", "tong", "tổng", "tiền", "tien", "amount", "pay"])

    def _is_table_question(self, question: str) -> bool:
        return any(keyword in question for keyword in ["bàn", "ban", "table"])

    def _is_items_question(self, question: str) -> bool:
        return any(
            keyword in question
            for keyword in [
                "món",
                "mon",
                "mặt hàng",
                "mat hang",
                "mua",
                "dùng gì",
                "dung gi",
                "ordered",
                "order",
                "items",
                "food",
                "drink",
            ]
        )
