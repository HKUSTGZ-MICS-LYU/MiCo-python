import os
import tempfile
import unittest

from MiCoEval import MiCoEval


def _build_eval(cache_path, cache_format, objective="latency_proxy"):
    evaluator = MiCoEval.__new__(MiCoEval)
    evaluator.output_json = cache_path
    evaluator.cache_format = cache_format
    evaluator.objective = objective
    evaluator.data_trace = {}
    evaluator.eval_f = lambda scheme: sum(scheme)
    return evaluator


class TestMiCoEvalCache(unittest.TestCase):
    def test_json_cache_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            evaluator = _build_eval(cache_path, "json")
            self.assertEqual(evaluator.eval([1, 2, 3]), 6)
            reloaded = _build_eval(cache_path, "json")
            reloaded.data_trace = reloaded._load_data_trace()
            self.assertEqual(reloaded.eval([1, 2, 3], offline=True), 6)

    def test_csv_cache_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.csv")
            evaluator = _build_eval(cache_path, "csv")
            self.assertEqual(evaluator.eval([4, 5]), 9)
            reloaded = _build_eval(cache_path, "csv")
            reloaded.data_trace = reloaded._load_data_trace()
            self.assertEqual(reloaded.eval([4, 5], offline=True), 9)

    def test_legacy_json_scheme_key_is_supported(self):
        evaluator = _build_eval("unused.json", "json")
        evaluator.data_trace = {"[1, 2]": {"latency_proxy": 3}}
        self.assertEqual(evaluator.eval([1, 2], offline=True), 3)
