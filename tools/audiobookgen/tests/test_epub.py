"""
Unit tests for EPUB parsing, kepub unwrapping, sentence span injection, SMIL building, and repacking.
"""

from pathlib import Path
import tempfile
import zipfile

from tools.audiobookgen.epub.kepub import KepubUnwrapper
from tools.audiobookgen.epub.markup import SentenceSpanInjector
from tools.audiobookgen.epub.package import EpubPackage
from tools.audiobookgen.epub.smil import SmilBuilder, format_smil_clock
from tools.audiobookgen.epub.writer import EpubWriter
from tools.audiobookgen.models import ChapterResult, SentenceTiming


def test_kepub_unwrapping():
    raw_kepub_snippet = (
        '<style type="text/css" class="kobostylehacks">div#book-inner { margin-top: 0;}</style>'
        '<p><st c="18"><span class="koboSpan" id="kobo.1.1">Tænk, at vi samler svampe, siger jeg, det har jeg aldrig prøvet før. </span></st>'
        '<st c="87"><span class="koboSpan" id="kobo.1.2">Sanker, siger Sebastian, man sanker. </span></st></p>'
    )
    clean = KepubUnwrapper.unwrap(raw_kepub_snippet)

    assert "koboSpan" not in clean
    assert "<st" not in clean
    assert "kobostylehacks" not in clean
    assert "Tænk, at vi samler svampe, siger jeg, det har jeg aldrig prøvet før. Sanker, siger Sebastian, man sanker." in clean


def test_sentence_span_injector():
    injector = SentenceSpanInjector()

    # Plain text paragraph
    html_plain = "<p>First sentence. Second sentence! Third sentence?</p>"
    marked, sents = injector.inject_spans(html_plain, chapter_index=1, lang="en")
    assert len(sents) == 3
    assert 'id="ch1_s1"' in marked
    assert 'id="ch1_s2"' in marked
    assert 'id="ch1_s3"' in marked
    assert sents[0] == ("ch1_s1", "First sentence.")

    # Inline container
    html_inline = "<p><em>Stop right there. Do not move.</em></p>"
    marked_inline, sents_inline = injector.inject_spans(html_inline, chapter_index=2, lang="en")
    assert len(sents_inline) == 2
    assert "ch2_s1" in marked_inline
    assert "ch2_s2" in marked_inline
    assert "<em>" in marked_inline

    # Idempotency test (processing already marked HTML should not double-wrap)
    re_marked, re_sents = injector.inject_spans(marked, chapter_index=1, lang="en")
    assert len(re_sents) == 3
    assert marked == re_marked


def test_smil_builder():
    timings = [
        SentenceTiming(span_id="ch1_s1", text="Hello world.", start=0.0, end=2.5),
        SentenceTiming(span_id="ch1_s2", text="How are you?", start=2.9, end=4.8),
    ]

    smil = SmilBuilder.build(
        timings=timings,
        xhtml_relative_path="../text/chapter01.xhtml",
        audio_relative_path="../Audio/ch_001.mp3",
    )

    assert '<smil xmlns="http://www.w3.org/ns/SMIL"' in smil
    assert '<seq epub:textref="../text/chapter01.xhtml">' in smil
    assert '<par id="par_ch1_s1">' in smil
    assert '<text src="../text/chapter01.xhtml#ch1_s1"/>' in smil
    assert '<audio src="../Audio/ch_001.mp3" clipBegin="00:00:00.000" clipEnd="00:00:02.500"/>' in smil
    assert '<par id="par_ch1_s2">' in smil
    assert '<audio src="../Audio/ch_001.mp3" clipBegin="00:00:02.900" clipEnd="00:00:04.800"/>' in smil


def test_epub_writer_and_package():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        source_epub = tmp_p / "source.epub"
        output_epub = tmp_p / "output.epub"

        # Create minimal valid source EPUB
        with zipfile.ZipFile(source_epub, "w") as zf:
            zf.writestr("mimetype", b"application/epub+zip", compress_type=zipfile.ZIP_STORED)
            zf.writestr(
                "META-INF/container.xml",
                '<?xml version="1.0"?>\n<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
                '  <rootfiles>\n    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>\n  </rootfiles>\n</container>',
            )
            opf = (
                '<?xml version="1.0" encoding="utf-8"?>\n'
                '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="pub-id">\n'
                '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
                '    <dc:title>Test Book</dc:title>\n'
                '    <dc:identifier id="pub-id">test-123</dc:identifier>\n'
                '    <dc:language>en</dc:language>\n'
                '  </metadata>\n'
                '  <manifest>\n'
                '    <item id="ch1" href="text/ch1.xhtml" media-type="application/xhtml+xml"/>\n'
                '  </manifest>\n'
                '  <spine>\n'
                '    <itemref idref="ch1"/>\n'
                '  </spine>\n'
                '</package>'
            )
            zf.writestr("OEBPS/content.opf", opf)
            xhtml = '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>Hello test.</p></body></html>'
            zf.writestr("OEBPS/text/ch1.xhtml", xhtml)

        # Parse with EpubPackage
        with EpubPackage(source_epub) as pkg:
            assert len(pkg.spine) == 1
            docs = list(pkg.iter_content_documents())
            assert len(docs) == 1
            assert docs[0][1] == "ch1"

            # Write audiobook with EpubWriter
            writer = EpubWriter(pkg)
            ch_res = ChapterResult(
                chapter_index=1,
                item_id="ch1",
                href="text/ch1.xhtml",
                marked_xhtml='<html xmlns="http://www.w3.org/1999/xhtml"><body><p><span id="ch1_s1">Hello test.</span></p></body></html>',
                smil_xml='<smil><par id="par_ch1_s1"><text src="../text/ch1.xhtml#ch1_s1"/><audio src="../Audio/ch_001.mp3" clipBegin="00:00:00.000" clipEnd="00:00:01.000"/></par></smil>',
                audio_bytes=b"FAKE_MP3_DATA",
                audio_mime="audio/mpeg",
                duration=1.0,
                timings=[SentenceTiming("ch1_s1", "Hello test.", 0.0, 1.0)],
            )
            out_f = writer.write_audiobook_epub(output_epub, [ch_res], total_duration=1.0)
            assert out_f.exists()

        # Validate output EPUB ZIP structure
        with zipfile.ZipFile(output_epub, "r") as out_zf:
            infolist = out_zf.infolist()
            # First file must be mimetype and uncompressed
            assert infolist[0].filename == "mimetype"
            assert infolist[0].compress_type == zipfile.ZIP_STORED

            # Check that audio file is ZIP_STORED
            audio_info = out_zf.getinfo("OEBPS/Audio/ch_001.mp3")
            assert audio_info.compress_type == zipfile.ZIP_STORED

            # Check that OPF contains media-overlay attributes
            out_opf = out_zf.read("OEBPS/content.opf").decode("utf-8")
            assert 'media-overlay="smil_ch1"' in out_opf
            assert 'id="smil_ch1"' in out_opf
            assert 'id="audio_ch1"' in out_opf
            assert '<meta property="media:duration">' in out_opf
