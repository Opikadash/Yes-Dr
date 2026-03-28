from app.text_utils import chunk_text, join_context, normalize_text


def test_normalize_text_strips_and_compacts():
    s = " hello\t\tworld \n\n\n\n next\x00line "
    out = normalize_text(s)
    assert "\x00" not in out
    assert "\n\n\n" not in out
    assert out.startswith("hello world")


def test_chunk_text_overlap():
    text = "a" * 50
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert chunks
    assert all(1 <= len(c) <= 20 for c in chunks)


def test_join_context_max_chars():
    ctx = join_context(["a" * 10, "b" * 10, "c" * 10], max_chars=15)
    assert len(ctx) <= 15

