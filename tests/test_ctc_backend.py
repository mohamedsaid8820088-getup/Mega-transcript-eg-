from mega_transcribe.ctc_backend import _decode_ids


def test_ctc_decoder_collapses_repeats_and_separates_words():
    tokens = {0: "<pad>", 4: "|", 12: "ا", 13: "ب", 14: "ة"}
    text, words = _decode_ids([12, 12, 0, 13, 4, 14, 14, 0], tokens)
    assert text == "ابة"
    assert [word for _, _, word in words] == ["اب", "ة"]
    assert words[0][0] == 0
    assert words[0][1] > words[0][0]


def test_ctc_decoder_ignores_blank_special_tokens_and_empty_words():
    tokens = {0: "<pad>", 1: "<s>", 2: "</s>", 3: "<unk>", 4: "|", 12: "ا"}
    text, words = _decode_ids([0, 4, 1, 12, 2, 0], tokens)
    assert text == "ا"
    assert [word for _, _, word in words] == ["ا"]
