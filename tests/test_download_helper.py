import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from helper_services.download_helper import download_files, download_zip_from_url, fetch_zip_files


class TestDownloadHelper(unittest.TestCase):
    def test_download_zip_from_url_uses_filename_from_url(self):
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.iter_content = Mock(return_value=[b"abc", b"123"])

        with tempfile.TemporaryDirectory() as download_dir:
            with patch("helper_services.download_helper.requests.get", return_value=mock_response):
                path = download_zip_from_url("https://example.com/my-data.zip", download_dir)

            self.assertTrue(path.endswith("my-data.zip"))
            self.assertTrue(os.path.exists(path))
            with open(path, "rb") as file:
                self.assertEqual(file.read(), b"abc123")

    def test_download_zip_from_url_uses_fallback_filename_for_non_zip_path(self):
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.iter_content = Mock(return_value=[b"x"])

        with tempfile.TemporaryDirectory() as download_dir:
            with patch("helper_services.download_helper.requests.get", return_value=mock_response):
                path = download_zip_from_url("https://example.com/not-a-zip", download_dir)

            basename = os.path.basename(path)
            self.assertTrue(basename.startswith("downloaded_"))
            self.assertTrue(basename.endswith(".zip"))
            self.assertTrue(os.path.exists(path))

    def test_fetch_zip_files_returns_sorted_downloaded_paths(self):
        with tempfile.TemporaryDirectory() as download_dir:
            with patch("helper_services.download_helper.atexit.register"), patch(
                "helper_services.download_helper.download_zip_from_url",
                side_effect=[
                    os.path.join(download_dir, "b.zip"),
                    os.path.join(download_dir, "a.zip"),
                ],
            ) as download_mock:
                result = fetch_zip_files(
                    ["https://x.com/b.zip", "https://x.com/a.zip"], download_dir
                )

        self.assertEqual(
            result,
            [os.path.join(download_dir, "a.zip"), os.path.join(download_dir, "b.zip")],
        )
        self.assertEqual(download_mock.call_count, 2)

    def test_download_files_without_urls_returns_none(self):
        self.assertIsNone(download_files([]))

    def test_download_files_with_urls_returns_directory_and_files(self):
        with patch("helper_services.download_helper.tempfile.gettempdir", return_value="/tmp"), patch(
            "helper_services.download_helper.fetch_zip_files",
            return_value=["/tmp/causal_analysis_fixed/a.zip"],
        ) as fetch_mock:
            download_dir, downloaded_files = download_files(["https://example.com/a.zip"])

        self.assertEqual(download_dir, "/tmp/causal_analysis_fixed")
        self.assertEqual(downloaded_files, ["/tmp/causal_analysis_fixed/a.zip"])
        fetch_mock.assert_called_once_with(
            ["https://example.com/a.zip"], "/tmp/causal_analysis_fixed"
        )


if __name__ == "__main__":
    unittest.main()
