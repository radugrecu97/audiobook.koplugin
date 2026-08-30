"""
End-to-end pipeline test generating a complete EPUB 3 audiobook with Media Overlays
using a FakeTTSProvider.
"""

from pathlib import Path
import tempfile
import zipfile

from tools.audiobookgen.audio.assembler import ChapterAudioAssembler
from tools.audiobookgen.audio.encoder import Mp3Encoder
from tools.audiobookgen.audio.pause import FixedPause
from tools.audiobookgen.audio.pcm import silence
from tools.audiobookgen.epub.kepub import KepubUnwrapper
from tools.audiobookgen.epub.markup import SentenceSpanInjector
from tools.audiobookgen.epub.package import EpubPackage
from tools.audiobookgen.models import AudioClip, SynthesisRequest, VoiceProfile
from tools.audiobookgen.pipeline.book import BookGenerationService
from tools.audiobookgen.pipeline.cache import DiskSentenceCache
from tools.audiobookgen.pipeline.chapter import ChapterSynthesisService
from tools.audiobookgen.qc.retry import RetryPolicy
from tools.audiobookgen.qc.verifier import NullVerifier
from tools.audiobookgen.text.normalizer import MultilingualNormalizer
from tools.audiobookgen.text.splitter import RegexSentenceSplitter
from tools.audiobookgen.tts.provider import TTSProvider


class FakeTTSProvider(TTSProvider):
    """Fake TTS Provider returning synthetic silence clips for tests."""

    def synthesize(self, request: SynthesisRequest) -> AudioClip:
        # Duration proportional to text length (0.05s per character, min 0.5s)
        dur = max(0.5, len(request.text) * 0.05)
        return silence(duration=dur, sample_rate=22050, channels=1, sample_width=2)


def create_sample_kepub(path: Path) -> Path:
    """Create a sample kepub file with 2 chapters."""
    with zipfile.ZipFile(path, "w") as zf:
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
            '    <dc:title>Sample Novel</dc:title>\n'
            '    <dc:identifier id="pub-id">book-12345</dc:identifier>\n'
            '    <dc:language>da</dc:language>\n'
            '  </metadata>\n'
            '  <manifest>\n'
            '    <item id="ch1" href="text/ch1.xhtml" media-type="application/xhtml+xml"/>\n'
            '    <item id="ch2" href="text/ch2.xhtml" media-type="application/xhtml+xml"/>\n'
            '  </manifest>\n'
            '  <spine>\n'
            '    <itemref idref="ch1"/>\n'
            '    <itemref idref="ch2"/>\n'
            '  </spine>\n'
            '</package>'
        )
        zf.writestr("OEBPS/content.opf", opf)

        ch1_html = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml">\n<body>\n'
            '<p><st c="1"><span class="koboSpan" id="kobo.1.1">Første sætning i bogen. </span></st>'
            '<st c="2"><span class="koboSpan" id="kobo.1.2">Anden sætning kommer her. </span></st></p>\n'
            '</body>\n</html>'
        )
        zf.writestr("OEBPS/text/ch1.xhtml", ch1_html)

        ch2_html = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml">\n<body>\n'
            '<p>Tredje sætning i andet kapitel. Fjerde sætning slutter bogen.</p>\n'
            '</body>\n</html>'
        )
        zf.writestr("OEBPS/text/ch2.xhtml", ch2_html)

    return path


def test_full_audiobook_generation_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        input_kepub = tmp_p / "sample.kepub"
        output_epub = tmp_p / "sample_audiobook.epub"
        work_dir = tmp_p / "work_dir"

        create_sample_kepub(input_kepub)

        cache = DiskSentenceCache(work_dir=work_dir)
        provider = FakeTTSProvider()
        verifier = NullVerifier()
        retry_policy = RetryPolicy(provider=provider, verifier=verifier, max_retries=0)
        pause_policy = FixedPause(duration=0.3)
        normalizer = MultilingualNormalizer()
        splitter = RegexSentenceSplitter(normalizer=normalizer)
        injector = SentenceSpanInjector(splitter=splitter)
        unwrapper = KepubUnwrapper()
        assembler = ChapterAudioAssembler(default_pause_policy=pause_policy)
        encoder = Mp3Encoder(bitrate="64k", channels=1, sample_rate=22050)

        chapter_service = ChapterSynthesisService(
            tts_provider=provider,
            retry_policy=retry_policy,
            cache=cache,
            encoder=encoder,
            pause_policy=pause_policy,
            normalizer=normalizer,
            splitter=splitter,
            injector=injector,
            unwrapper=unwrapper,
            assembler=assembler,
        )

        book_service = BookGenerationService(
            chapter_service=chapter_service,
            cache=cache,
        )

        voice = VoiceProfile(temperature=0.75, seed=42)

        out_path = book_service.run(
            input_epub=input_kepub,
            output_epub=output_epub,
            voice=voice,
            lang="da",
            resume=True,
        )

        assert out_path is not None
        assert out_path.exists()

        # Verify output EPUB ZIP
        with zipfile.ZipFile(out_path, "r") as zf:
            # 1. mimetype check
            infolist = zf.infolist()
            assert infolist[0].filename == "mimetype"
            assert infolist[0].compress_type == zipfile.ZIP_STORED

            # 2. SMIL files
            assert "OEBPS/MediaOverlays/ch_001.smil" in zf.namelist()
            assert "OEBPS/MediaOverlays/ch_002.smil" in zf.namelist()

            smil1 = zf.read("OEBPS/MediaOverlays/ch_001.smil").decode("utf-8")
            assert 'epub:textref="../text/ch1.xhtml"' in smil1
            assert '<par id="par_ch1_s1">' in smil1
            assert '<text src="../text/ch1.xhtml#ch1_s1"/>' in smil1
            assert '<audio src="../Audio/ch_001.mp3"' in smil1

            # 3. Audio files
            assert "OEBPS/Audio/ch_001.mp3" in zf.namelist()
            assert "OEBPS/Audio/ch_002.mp3" in zf.namelist()
            audio_entry = zf.getinfo("OEBPS/Audio/ch_001.mp3")
            assert audio_entry.compress_type == zipfile.ZIP_STORED

            # 4. OPF manifest & metadata
            opf_text = zf.read("OEBPS/content.opf").decode("utf-8")
            assert 'media-overlay="smil_ch1"' in opf_text
            assert 'media-overlay="smil_ch2"' in opf_text
            assert '<item id="smil_ch1" href="MediaOverlays/ch_001.smil" media-type="application/smil+xml"/>' in opf_text
            assert '<item id="audio_ch1" href="Audio/ch_001.mp3" media-type="audio/mpeg"/>' in opf_text
            assert '<meta property="media:duration">' in opf_text
            assert '<meta property="media:active-class">-epub-media-overlay-active</meta>' in opf_text

            # 5. Marked XHTML
            ch1_marked = zf.read("OEBPS/text/ch1.xhtml").decode("utf-8")
            assert 'id="ch1_s1"' in ch1_marked
            assert 'id="ch1_s2"' in ch1_marked
            assert "koboSpan" not in ch1_marked
