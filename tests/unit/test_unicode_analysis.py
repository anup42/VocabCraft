from vocabcraft.unicode_analysis import analyze_piece, visible_piece


def test_sentencepiece_whitespace_marker_handling() -> None:
    analysis = analyze_piece("▁Hello")
    assert visible_piece("▁Hello") == "Hello"
    assert analysis.visible_text == "Hello"
    assert analysis.has_sentencepiece_whitespace
    assert analysis.unicode_scripts == {"Latin"}


def test_devanagari_script_analysis() -> None:
    analysis = analyze_piece("▁फ़ोन")
    assert analysis.unicode_scripts == {"Devanagari"}
    assert {"Lo", "Mn", "Mc"} <= analysis.unicode_categories


def test_common_and_inherited_categories() -> None:
    punctuation = analyze_piece("!?₹")
    combining_mark = analyze_piece("\u0301")
    assert punctuation.unicode_scripts == {"Common"}
    assert {"Po", "Sc"} <= punctuation.unicode_categories
    assert combining_mark.unicode_scripts == {"Inherited"}
    assert combining_mark.unicode_categories == {"Mn"}


def test_mixed_latin_devanagari_piece() -> None:
    analysis = analyze_piece("▁दिल्लीMetro")
    assert analysis.unicode_scripts == {"Latin", "Devanagari"}
