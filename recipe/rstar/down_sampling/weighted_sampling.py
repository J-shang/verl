import re
import numpy as np
import torch

from verl.protocol import DataProto
from .utils import filter_by_mask


def weighted_sampling(batch: DataProto, do_sample=True, world_size=None):
    pass


def error_ratio_weighted_sampling(batch: DataProto, do_sample=True, world_size=None):
    def error_ratio(index, text, pattern=r'<tool_response>.*?</tool_response>'):
        matches = re.findall(pattern, text, re.DOTALL)
        error_count = len([match for match in matches if 'error' in match.lower()])
        if len(matches) == 0:
            return index, 0.5
        else:
            return index, error_count / len(matches)
