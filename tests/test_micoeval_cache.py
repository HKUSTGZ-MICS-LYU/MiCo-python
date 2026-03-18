import os
import tempfile
import unittest
from multiprocessing import Process

from MiCoEval import MiCoEval


def _build_eval(cache_path, cache_format, objective="latency_proxy"):
    evaluator = MiCoEval.__new__(MiCoEval)
    evaluator.output_json = cache_path
    evaluator.cache_format = cache_format
    evaluator.objective = objective
    evaluator.data_trace = {}
    evaluator.eval_f = lambda scheme: sum(scheme)
    return evaluator


def _concurrent_eval_worker(cache_path, schemes):
    evaluator = _build_eval(cache_path, "json")
    for scheme in schemes:
        evaluator.eval(scheme)


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
        self.assertIn("1,2", evaluator.data_trace)
        self.assertNotIn("[1, 2]", evaluator.data_trace)

    def test_offline_mode_raises_on_cache_miss(self):
        evaluator = _build_eval("unused.json", "json")
        with self.assertRaises(ValueError):
            evaluator.eval([9, 9], offline=True)

    def test_json_cache_concurrent_process_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            schemes_a = [[i, i + 1] for i in range(0, 80, 2)]
            schemes_b = [[i, i + 1] for i in range(1, 81, 2)]

            p1 = Process(target=_concurrent_eval_worker, args=(cache_path, schemes_a))
            p2 = Process(target=_concurrent_eval_worker, args=(cache_path, schemes_b))
            p1.start()
            p2.start()
            p1.join()
            p2.join()

            self.assertEqual(p1.exitcode, 0)
            self.assertEqual(p2.exitcode, 0)

            reloaded = _build_eval(cache_path, "json")
            reloaded.data_trace = reloaded._load_data_trace()
            for scheme in schemes_a + schemes_b:
                self.assertEqual(reloaded.eval(scheme, offline=True), sum(scheme))
