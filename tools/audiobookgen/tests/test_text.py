"""
Unit tests for text normalization and sentence splitting across English, Danish, and Hungarian.
"""

from tools.audiobookgen.text.normalizer import MultilingualNormalizer
from tools.audiobookgen.text.splitter import RegexSentenceSplitter


def test_danish_normalization():
    norm = MultilingualNormalizer()

    # Abbreviations
    text = "Det er f.eks. bl.a. dvs. og osv. samt ca. 5 stk. til 100 kr. hos dr. Hansen og prof. Møller."
    result = norm.normalize(text, lang="da")
    assert "for eksempel" in result
    assert "blandt andet" in result
    assert "det vil sige" in result
    assert "og så videre" in result
    assert "cirka" in result
    assert "stykker" in result
    assert "kroner" in result
    assert "doktor" in result
    assert "professor" in result

    # Additional Danish abbreviations
    extra = "m.fl. og d.v.s. samt pga. og p.g.a. hr. Nielsen og fr. Jensen."
    res_extra = norm.normalize(extra, lang="da")
    assert "med flere" in res_extra
    assert "det vil sige" in res_extra
    assert "på grund af" in res_extra
    assert "herre" in res_extra
    assert "frue" in res_extra

    # Danish decimals (comma and period)
    assert norm.normalize("58,6 procent", lang="da") == "58 komma 6 procent"
    assert norm.normalize("12.5 kroner", lang="da") == "12 komma 5 kroner"
    assert norm.normalize("100%", lang="da") == "100 procent"


def test_hungarian_normalization():
    norm = MultilingualNormalizer()

    text = "A Duna Aszfalt Zrt. és a Kft. prof. Nagy és dr. Kiss vezetésével 58,6 milliárd Ft összeget kapott, ami 15%-os növekedés."
    result = norm.normalize(text, lang="hu")
    assert "Zrt" in result
    assert "Kft" in result
    assert "professzor" in result
    assert "doktor" in result
    assert "58 egész 6 tized" in result
    assert "forint" in result
    assert "százalék" in result


def test_english_normalization():
    norm = MultilingualNormalizer()

    text = "Dr. Smith and Mr. Brown met Prof. Davis at St. John's, approx. 12.5 miles away, etc."
    result = norm.normalize(text, lang="en")
    assert "Doctor" in result
    assert "Mister" in result
    assert "Professor" in result
    assert "Saint" in result
    assert "approximately" in result
    assert "12 point 5" in result
    assert "etcetera" in result


def test_sentence_splitting_multilingual():
    splitter = RegexSentenceSplitter()

    # English standard sentences with quotes
    en_text = 'She said, "Hello there!" He replied, "Good morning." Then they walked away.'
    en_sents = splitter.split(en_text, lang="en")
    assert len(en_sents) == 3

    # Danish sentences with abbreviations that should not break mid-sentence
    da_text = "Danmarks Nationalbank vurderer, at den økonomiske vækst fortsætter f.eks. i 2026. Dette skyldes øget privatforbrug."
    da_sents = splitter.split(da_text, lang="da")
    assert len(da_sents) == 2
    assert "for eksempel" in da_sents[0]

    # Speaker tag turns
    speaker_text = "<|speaker:0|>Turn one.\n<|speaker:1|>Turn two."
    s_sents = splitter.split(speaker_text, lang="en")
    assert len(s_sents) == 2
    assert s_sents[0] == "<|speaker:0|>Turn one."
    assert s_sents[1] == "<|speaker:1|>Turn two."


def test_max_chunk_cap():
    splitter = RegexSentenceSplitter(max_chunk_chars=50)

    # A very long continuous sentence without standard sentence end marks
    long_sent = "This is a very long run-on sentence, which contains multiple clauses, and should be split cleanly into smaller parts without hallucinating."
    chunks = splitter.split(long_sent, lang="en")
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 60  # Allows slight boundary buffer
