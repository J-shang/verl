import numpy as np
import torch

from verl.protocol import DataProto
from .utils import filter_by_mask


def reject_equal_reward(batch: DataProto, do_sample=True, world_size=None):
    # Rejection sampling based on rewards
    # Group rewards by uid
    uids = batch.non_tensor_batch['uid']
    unique_uids = np.unique(uids)
    valid_mask = torch.ones(len(uids), dtype=torch.bool)
    solve_equal = 0
    solve_equal_zeros = 0
    solve_equal_non_all_zeros = 0

    for uid in unique_uids:
        uid_mask = uids == uid
        # Sum rewards for each sequence
        uid_rewards = batch.batch['token_level_scores'][uid_mask].sum(-1)

        if torch.allclose(uid_rewards[0], uid_rewards):
            valid_mask[uid_mask] = False
            solve_equal += 1

        if torch.allclose(torch.zeros_like(uid_rewards), uid_rewards):
            solve_equal_zeros += 1
        else:
            solve_equal_non_all_zeros += 1

    metrics = {}
    metrics['batch/solve_equal'] = solve_equal
    metrics['batch/reward_all_zeros'] = solve_equal_zeros
    metrics['batch/reward_non_all_zeros'] = solve_equal_non_all_zeros

    if do_sample:
        if not valid_mask.any():
            return None, metrics
        batch = filter_by_mask(batch, valid_mask, world_size)
        if batch is None:
            return None, metrics
        
