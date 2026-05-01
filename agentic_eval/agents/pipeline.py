from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING, Any, Callable

from agentic_eval.agents.diagnosis_agent import DiagnosisAgent
from agentic_eval.agents.evaluation_agent import EvaluationAgent
from agentic_eval.schemas.case_schema import EvaluationCase
from agentic_eval.schemas.result_schema import EvaluationResult

if TYPE_CHECKING:
    pass

_SENTINEL = object()


class EvalPipeline:
    """
    真正的 Multi-Agent 流水线：EvaluationAgent 和 DiagnosisAgent 作为独立线程并发运行，
    通过队列传递消息。某个 case 完成 eval 后立刻进入 diagnosis，而无需等待全部 eval 结束。

    拓扑：
        [eval workers ×N]  --eval_queue-->  [diagnosis workers ×M]  --done_queue-->  [collector]
    """

    def __init__(
        self,
        evaluator: EvaluationAgent,
        diagnoser: DiagnosisAgent,
        eval_workers: int = 2,
        diagnosis_workers: int = 1,
        llm_failed_only: bool = True,
        on_eval_done: Callable[[EvaluationResult], None] | None = None,
        on_diagnosis_done: Callable[[EvaluationResult], None] | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.diagnoser = diagnoser
        self.eval_workers = max(1, eval_workers)
        self.diagnosis_workers = max(1, diagnosis_workers)
        self.llm_failed_only = llm_failed_only
        self.on_eval_done = on_eval_done
        self.on_diagnosis_done = on_diagnosis_done

    def run(self, cases: list[EvaluationCase]) -> list[EvaluationResult]:
        """
        启动流水线，返回按原始 case 顺序排列的 EvaluationResult 列表。
        """
        if not cases:
            return []

        # eval_queue: CaseGeneratorAgent 的输出，每项为 (original_index, EvaluationCase)
        # diagnosis_queue: EvaluationAgent 的输出，每项为 EvaluationResult 或 _SENTINEL
        eval_queue: queue.Queue[tuple[int, EvaluationCase] | object] = queue.Queue()
        diagnosis_queue: queue.Queue[EvaluationResult | object] = queue.Queue()
        done_queue: queue.Queue[EvaluationResult] = queue.Queue()

        errors: list[Exception] = []
        errors_lock = threading.Lock()

        # --- 生产者：把所有 case 入队 ---
        for idx, case in enumerate(cases):
            eval_queue.put((idx, case))
        for _ in range(self.eval_workers):
            eval_queue.put(_SENTINEL)

        # --- EvaluationAgent 工作线程 ---
        def eval_worker() -> None:
            while True:
                item = eval_queue.get()
                if item is _SENTINEL:
                    eval_queue.task_done()
                    break
                idx, case = item  # type: ignore[misc]
                try:
                    result = self.evaluator.evaluate_case(case)
                    result._pipeline_idx = idx  # type: ignore[attr-defined]
                    if self.on_eval_done:
                        self.on_eval_done(result)
                    diagnosis_queue.put(result)
                except Exception as exc:
                    with errors_lock:
                        errors.append(exc)
                finally:
                    eval_queue.task_done()

        # --- DiagnosisAgent 工作线程 ---
        def diagnosis_worker() -> None:
            while True:
                item = diagnosis_queue.get()
                if item is _SENTINEL:
                    diagnosis_queue.task_done()
                    break
                result: EvaluationResult = item  # type: ignore[assignment]
                try:
                    if result.passed and self.llm_failed_only:
                        self.diagnoser.diagnose_with_rules(result)
                    else:
                        self.diagnoser.diagnose(result)
                    if self.on_diagnosis_done:
                        self.on_diagnosis_done(result)
                    done_queue.put(result)
                except Exception as exc:
                    with errors_lock:
                        errors.append(exc)
                    done_queue.put(result)
                finally:
                    diagnosis_queue.task_done()

        # --- 启动所有工作线程 ---
        eval_threads = [
            threading.Thread(target=eval_worker, daemon=True, name=f"eval-{i}")
            for i in range(self.eval_workers)
        ]
        diagnosis_threads = [
            threading.Thread(target=diagnosis_worker, daemon=True, name=f"diag-{i}")
            for i in range(self.diagnosis_workers)
        ]
        for t in eval_threads + diagnosis_threads:
            t.start()

        # --- 等待所有 eval 完成，然后向 diagnosis 发哨兵 ---
        def _wait_eval_then_signal() -> None:
            for t in eval_threads:
                t.join()
            for _ in range(self.diagnosis_workers):
                diagnosis_queue.put(_SENTINEL)

        signal_thread = threading.Thread(target=_wait_eval_then_signal, daemon=True)
        signal_thread.start()

        # --- 等待所有 diagnosis 完成 ---
        for t in diagnosis_threads:
            t.join()
        signal_thread.join()

        # --- 收集结果，按原始顺序排列 ---
        results: list[Any] = [None] * len(cases)
        while not done_queue.empty():
            result = done_queue.get_nowait()
            idx = getattr(result, "_pipeline_idx", None)
            if idx is not None:
                results[idx] = result

        if errors:
            raise RuntimeError(
                f"Pipeline encountered {len(errors)} error(s). First: {errors[0]}"
            ) from errors[0]

        return [r for r in results if r is not None]

    def run_with_progress(
        self,
        cases: list[EvaluationCase],
        on_eval_done: Callable[[EvaluationResult], None] | None = None,
        on_diagnosis_done: Callable[[EvaluationResult], None] | None = None,
    ) -> list[EvaluationResult]:
        """run() 的别名，支持在调用时临时覆盖回调（方便 run_eval.py 注入进度更新）。"""
        orig_eval = self.on_eval_done
        orig_diag = self.on_diagnosis_done
        if on_eval_done:
            self.on_eval_done = on_eval_done
        if on_diagnosis_done:
            self.on_diagnosis_done = on_diagnosis_done
        try:
            return self.run(cases)
        finally:
            self.on_eval_done = orig_eval
            self.on_diagnosis_done = orig_diag
