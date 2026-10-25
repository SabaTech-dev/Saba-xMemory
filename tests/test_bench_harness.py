"""Unit tests for the benchmark harness (benchmarks/common.py).

The harness is not a package; tests import it by path. These tests pin the
error-accounting and retry behavior that makes benchmark runs auditable:
a run polluted by LLM-endpoint errors must surface them in the summary and
must retry errored items before giving up on them.
"""

import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from common import BenchRunner, summarize  # noqa: E402


class TestSummarizeErrorAccounting:
    def test_summary_counts_answer_and_judge_errors(self):
        records = [
            {"correct": True, "category": "a", "predicted": "x"},
            {"correct": False, "category": "a", "predicted": "__ERROR__: boom"},
            {"correct": False, "category": "b", "predicted": "y", "judge_error": True},
        ]
        s = summarize(records)
        assert s["overall"]["answer_errors"] == 1
        assert s["overall"]["judge_errors"] == 1
        # errors count as incorrect (conservative), accuracy unchanged in form
        assert s["overall"]["accuracy"] == round(1 / 3 * 100, 2)

    def test_summary_zero_errors_when_clean(self):
        records = [
            {"correct": True, "category": "a", "predicted": "x"},
            {"correct": False, "category": "a", "predicted": "nope"},
        ]
        s = summarize(records)
        assert s["overall"]["answer_errors"] == 0
        assert s["overall"]["judge_errors"] == 0


class TestRunQARetrySweep:
    @staticmethod
    def _runner() -> BenchRunner:
        runner = BenchRunner.__new__(BenchRunner)
        runner.bank_id = "unit-test"
        return runner

    def test_errored_answer_is_retried_before_recording(self):
        runner = self._runner()
        calls = {"n": 0}

        def fake_answer(question, options, top_k=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"answer": "__ERROR__: transient 500", "retrieved": [], "latency_ms": 1}
            return {"answer": "good answer", "retrieved": ["m1"], "latency_ms": 1}

        runner.answer_question = fake_answer
        runner.judge = lambda question, reference, predicted: (True, False)

        records, _ = runner.run_qa(
            [{"question": "q?", "answer": "good answer"}], concurrency=1, progress_every=0
        )
        assert calls["n"] == 2  # one failing attempt + one sweep retry
        assert records[0]["predicted"] == "good answer"
        assert records[0]["correct"] is True
        assert records[0]["error"] is False

    def test_persistent_errors_are_flagged_not_hidden(self):
        runner = self._runner()

        def always_error(question, options, top_k=None):
            return {"answer": "__ERROR__: down", "retrieved": [], "latency_ms": 1}

        runner.answer_question = always_error
        runner.judge = lambda question, reference, predicted: (False, False)

        records, _ = runner.run_qa(
            [{"question": "q?", "answer": "a"}], concurrency=1, progress_every=0
        )
        assert records[0]["error"] is True
        assert records[0]["correct"] is False  # conservative: error != correct

    def test_judge_call_error_is_flagged_and_retried(self):
        runner = self._runner()
        judge_calls = {"n": 0}

        def good_answer(question, options, top_k=None):
            return {"answer": "paris", "retrieved": ["m"], "latency_ms": 1}

        def flaky_judge(question, reference, predicted):
            judge_calls["n"] += 1
            if judge_calls["n"] == 1:
                return (False, True)  # judge endpoint failed
            return (True, False)

        runner.answer_question = good_answer
        runner.judge = flaky_judge

        records, _ = runner.run_qa(
            [{"question": "capital?", "answer": "paris"}], concurrency=1, progress_every=0
        )
        assert judge_calls["n"] >= 2
        assert records[0]["judge_error"] is False
        assert records[0]["correct"] is True
