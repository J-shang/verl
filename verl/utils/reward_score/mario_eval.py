from typing import Optional, List, Union

from math_evaluation import (
    string_normalize,
    are_equal_under_sympy,
    is_equiv,
    is_equiv_type,
)

import re
from .qwen_math_eval_toolkit.parser import extract_answer


def compute_score(solution_str, ground_truth) -> float:
    # breakpoint()
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL,count = 1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    extracted_answer = extract_answer(model_output, data_name="math")  #TODO: check the data_name, hard code here for now

    if is_equiv_MATH(ground_truth=ground_truth, prediction=extracted_answer, verbose=False):
        box_match = 1.0
    else:
        box_match = -0.5

    if "boxed" not in model_output:
        box_match = -1.0

    return box_match


def is_equiv_MATH(
    ground_truth: Union[str, List[str]],
    prediction: str,
    verbose: bool = False,
) -> bool:
    """
    We manually annotated the ground-truth of MATH testset.

    For some questions, one solution of the provided answers is also considered as correct.
    """
    if isinstance(ground_truth, list):
        for grt in ground_truth:
            if is_equiv(grt, prediction, verbose):
                return True
        return False
    else:
        assert isinstance(ground_truth, str)
        return is_equiv(ground_truth, prediction, verbose)
