import importlib
import math
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class DummyFrame:
    def __getitem__(self, _):
        return self

    def copy(self):
        return self


class TestLambdaHandler(unittest.TestCase):
    def _import_lambda_module_with_stubs(self):
        fake_causalbench = types.ModuleType("causalbench")
        fake_causalbench.services = types.SimpleNamespace(
            auth=types.SimpleNamespace(__access_token=None)
        )

        fake_numpy = types.ModuleType("numpy")
        fake_numpy.sqrt = math.sqrt

        fake_analysis_module = types.ModuleType("helper_services.causal_analysis_helper")
        fake_analysis_module.run_causal_analysis = lambda *args, **kwargs: ({}, "/tmp")

        fake_reco_module = types.ModuleType("helper_services.causal_recommendation_helper")
        fake_reco_module.run_causal_recommendation = lambda *args, **kwargs: []

        fake_g2s_reco_module = types.ModuleType(
            "helper_services.g2s_causal_recommendation_helper"
        )
        fake_g2s_reco_module.run_g2s_causal_recommendation = (
            lambda *args, **kwargs: []
        )

        fake_download_module = types.ModuleType("helper_services.download_helper")
        fake_download_module.download_files = lambda *args, **kwargs: ("/tmp", [])

        fake_report_module = types.ModuleType("helper_services.report_helper")
        fake_report_module.generate_report = lambda *args, **kwargs: ("a.yml", "a.pdf", "a.xlsx")

        fake_hp_dtype_module = types.ModuleType("helper_services.hp_dtype_helper")
        fake_hp_dtype_module.get_hp_dtypes = lambda *args, **kwargs: {}

        fake_mail_module = types.ModuleType("helper_services.mail_helper")
        fake_mail_module.send_email = lambda *args, **kwargs: {"status": "ok"}

        stub_modules = {
            "causalbench": fake_causalbench,
            "numpy": fake_numpy,
            "helper_services.causal_analysis_helper": fake_analysis_module,
            "helper_services.causal_recommendation_helper": fake_reco_module,
            "helper_services.g2s_causal_recommendation_helper": fake_g2s_reco_module,
            "helper_services.download_helper": fake_download_module,
            "helper_services.report_helper": fake_report_module,
            "helper_services.hp_dtype_helper": fake_hp_dtype_module,
            "helper_services.mail_helper": fake_mail_module,
        }

        sys.modules.pop("lambda_function", None)
        with patch.dict(sys.modules, stub_modules):
            module = importlib.import_module("lambda_function")
        return module

    def test_handler_returns_analysis_and_sends_attachments(self):
        lambda_module = self._import_lambda_module_with_stubs()
        causal_results = {
            "Metric.Score": {
                "effects": {"HP.min_samples_leaf": 0.8, "HP.max_features": 0},
                "data": DummyFrame(),
                "recommendations": [],
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, "out.pdf")
            xlsx_path = os.path.join(temp_dir, "out.xlsx")
            with open(pdf_path, "wb"):
                pass
            with open(xlsx_path, "wb"):
                pass

            with patch.object(
                lambda_module, "download_files", return_value=("/tmp/download", ["one.zip"])
            ) as download_mock, patch.object(
                lambda_module, "get_hp_dtypes", return_value={"min_samples_leaf": "integer"}
            ) as dtypes_mock, patch.object(
                lambda_module, "run_causal_analysis", return_value=(causal_results, "/tmp/download")
            ) as analysis_mock, patch.object(
                lambda_module, "run_g2s_causal_recommendation", return_value=[{"delta": 1}]
            ) as reco_mock, patch.object(
                lambda_module, "generate_report", return_value=("out.yml", pdf_path, xlsx_path)
            ) as report_mock, patch.object(
                lambda_module, "send_email", return_value={"status": "ok"}
            ) as email_mock:
                event = {
                    "zip_urls": ["https://example.com/a.zip"],
                    "outcome_column": "Metric.Score",
                    "candidate_hyperparameters": [],
                    "hyperparameter_limits": {"min_samples_leaf": {"min": 1, "max": 10}},
                    "user_email": "user@example.com",
                    "unique_id": "abc123",
                    "run_ids": [1],
                    "filters": {"Run ID": ["1"]},
                    "jwt_token": "token-123",
                }
                response = lambda_module.handler(event, context={})

        self.assertIn("analysis_results", response)
        self.assertIn("Metric.Score", response["analysis_results"])
        self.assertEqual(
            getattr(lambda_module.causalbench.services.auth, "__access_token"),
            "token-123",
        )
        self.assertTrue(os.environ["HOME"].endswith("/tmp/home"))
        self.assertTrue(os.environ["USERPROFILE"].endswith("/tmp/home"))

        download_mock.assert_called_once()
        dtypes_mock.assert_called_once_with("/tmp/download")
        analysis_mock.assert_called_once()
        reco_mock.assert_called_once()
        report_mock.assert_called_once()
        email_mock.assert_called_once()

        email_args = email_mock.call_args.args
        email_kwargs = email_mock.call_args.kwargs
        self.assertEqual(email_args[0], "user@example.com")
        self.assertIn(pdf_path, email_kwargs["attachments"])
        self.assertIn(xlsx_path, email_kwargs["attachments"])


if __name__ == "__main__":
    unittest.main()
