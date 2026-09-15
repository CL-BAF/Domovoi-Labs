"""Fast research evaluation suite (see EVALS.md section 1).

Implemented now:  val_ppl, HellaSwag, ARC-Challenge, PIQA, WinoGrande,
GSM8K subset, IF mini-suite.

Slots reserved for milestone evals (same interface, registered in
domovoi/evals/base.py TASKS):  MMLU-Pro, GPQA-Diamond, MATH-500,
LiveCodeBench, IFBench, RULER.
"""

from .base import (DEFAULT_TASKS, EVAL_DATA_ROOT, TASKS, MultipleChoiceEval,
                   build_task, choice_logprob, read_eval_data, run_evals)
from .gsm8k import GSM8KSubsetEval
from .if_mini import IFMiniEval
from .val_ppl import ValPPLEval

__all__ = [
    "DEFAULT_TASKS", "EVAL_DATA_ROOT", "TASKS", "MultipleChoiceEval",
    "build_task", "choice_logprob", "read_eval_data", "run_evals",
    "GSM8KSubsetEval", "IFMiniEval", "ValPPLEval",
]