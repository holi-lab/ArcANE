"""Offline tests for the Project Gutenberg source downloader."""

from contextlib import redirect_stderr, redirect_stdout
from email.message import Message
from io import BytesIO, StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from arc_construction import download_novels as downloader


def source_text(ebook_id=1399, marker="THE"):
    return (
        f"The Project Gutenberg eBook of Example\n[EBook #{ebook_id}]\n"
        f"*** START OF {marker} PROJECT GUTENBERG EBOOK EXAMPLE ***\n"
        + "A paragraph of synthetic test content.\n" * 100
        + f"*** END OF {marker} PROJECT GUTENBERG EBOOK EXAMPLE ***\n"
    ).encode("utf-8")


def response(content, content_type="text/plain"):
    result = BytesIO(content)
    result.headers = Message()
    result.headers["Content-Type"] = content_type
    return result


class SourceDownloadTests(unittest.TestCase):
    def test_catalog_matches_public_artifact_slugs(self):
        release = Path(__file__).resolve().parents[1]
        slugs = {path.name for path in (release / "results/arc_extraction").iterdir()
                 if path.is_dir()}
        self.assertEqual(set(downloader.SOURCES), slugs)
        self.assertEqual(len(downloader.SOURCES), 18)
        self.assertNotIn("Harry_Potter", downloader.SOURCES)

    def test_accepts_modern_legacy_and_bom_text(self):
        for marker in ("THE", "THIS"):
            with self.subTest(marker=marker):
                downloader.validate_text(source_text(marker=marker), 1399)
                downloader.validate_text(b"\xef\xbb\xbf" + source_text(marker=marker), 1399)

    def test_rejects_wrong_edition_truncation_html_and_encoding(self):
        invalid = [source_text(1184), b"<html>Error</html>", b"\xff\xfe"]
        invalid.append(source_text().split(b"*** END")[0])
        invalid.append(source_text().replace(b"A paragraph of synthetic test content.\n", b""))
        for content in invalid:
            with self.subTest(content_length=len(content)), self.assertRaises(ValueError):
                downloader.validate_text(content, 1399)

    def test_rejects_oversized_content(self):
        with patch.object(downloader, "MAX_DOWNLOAD_BYTES", 10):
            with self.assertRaises(ValueError):
                downloader.validate_text(source_text(), 1399)

    def test_saves_original_bytes_and_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(downloader, "urlopen", return_value=response(source_text())) as fetch:
                status = downloader.download_novel("Anna_Kareina", output)
            self.assertEqual((output / "Anna_Kareina.txt").read_bytes(), source_text())
            self.assertEqual(len(list(output.iterdir())), 1)
            self.assertIn("sha256=", status)
            self.assertEqual(fetch.call_args.args[0].full_url,
                             f"{downloader.MIRROR}/1399/pg1399.txt")

    def test_existing_file_is_unchanged_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            destination = output / "Anna_Kareina.txt"
            destination.write_bytes(b"existing local copy")
            with patch.object(downloader, "urlopen") as fetch:
                self.assertTrue(downloader.download_novel("Anna_Kareina", output).startswith("KEEP"))
            fetch.assert_not_called()
            self.assertEqual(destination.read_bytes(), b"existing local copy")

    def test_failed_download_never_creates_destination(self):
        for result in (response(b"error", "text/html"), response(source_text(1184))):
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(downloader, "urlopen", return_value=result):
                    with self.assertRaises(ValueError):
                        downloader.download_novel("Anna_Kareina", Path(directory))
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_write_race_does_not_overwrite_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            destination = output / "Anna_Kareina.txt"

            def another_writer_wins(_source, target):
                Path(target).write_bytes(b"other writer")
                raise FileExistsError(target)

            with patch.object(downloader, "urlopen", return_value=response(source_text())):
                with patch.object(downloader.os, "link", side_effect=another_writer_wins):
                    with self.assertRaises(FileExistsError):
                        downloader.download_novel("Anna_Kareina", output)
            self.assertEqual(destination.read_bytes(), b"other writer")
            self.assertEqual(len(list(output.iterdir())), 1)

    def test_cli_list_is_offline_and_failures_return_nonzero(self):
        with patch.object(downloader, "urlopen") as fetch, redirect_stdout(StringIO()):
            self.assertEqual(downloader.main(["--list"]), 0)
            fetch.assert_not_called()
        with patch.object(downloader, "download_novel", side_effect=URLError("offline")):
            with redirect_stderr(StringIO()):
                self.assertEqual(downloader.main(["--novel", "Anna_Kareina"]), 1)

    def test_cli_validates_numbers_and_unknown_sources(self):
        invalid = [
            ["--novel", "Harry_Potter"],
            ["--list", "--timeout", "nan"],
            ["--list", "--timeout", "0"],
            ["--list", "--delay", "-1"],
            ["--list", "--delay", "inf"],
        ]
        for args in invalid:
            with self.subTest(args=args), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    downloader.main(args)
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
