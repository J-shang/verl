import re
import numpy as np
import torch
from typing import List
from transformers import PreTrainedTokenizerFast

from verl.protocol import DataProto
from .utils import filter_by_mask, decode_prompt_response_str


def fused_weighted_sampling(batch: DataProto, tokenizer: PreTrainedTokenizerFast, config: dict, world_size=None):
    do_error_ratio_weighted = config.get("error_ratio_weighted", False)

    _, response_text = decode_prompt_response_str(batch, tokenizer)
    penalty_weights = np.zeros(len(response_text))
    metrics = {}

    # calculate error ratio weight
    _penalty_weights, _metrics = calc_error_ratio_weight(response_text)
    metrics.update(_metrics)
    if do_error_ratio_weighted:
        penalty_weights += _penalty_weights

    

def calc_error_ratio_weight(response_text: List[str]):
    def error_ratio(text, pattern=r'<tool_response>.*?</tool_response>'):
        matches = re.findall(pattern, text, re.DOTALL)
        error_count = len([match for match in matches if 'error' in match.lower()])
        if len(matches) == 0:
            return 0.5, 0, 0
        else:
            return error_count / len(matches), error_count, len(matches)

    penalty_weights, metrics = [], {}
    total_error_count, total_res_count = 0, 0

    for text in response_text:
        penalty_weight, error_count, res_count = error_ratio(text) 
        penalty_weights.append(penalty_weight)
        total_error_count += error_count
        total_res_count += res_count
    metrics = {
        'error_ratio_weighted_sampling/global_err_ratio': total_error_count / total_res_count if total_res_count > 0 else 0,
        'error_ratio_weighted_sampling/penalty_weight': np.mean(penalty_weights) if penalty_weights else 0,
    }
    return np.array(penalty_weights), metrics
