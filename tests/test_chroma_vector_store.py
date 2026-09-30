from src.infrastructure.vector_store.chroma_vector_store import ChromaVectorStore


def test_chroma_vector_store_indexes_and_searches(tmp_path):
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma")
    store.index(
        document_id="doc-1",
        text="MINI MART AN PHU\nNgay ban: 12/08/2025\nTong cong: 92000 VND",
    )

    results = store.search(document_id="doc-1", query="tong tien hoa don", limit=2)

    assert results
    assert "Tong cong" in results[0]

