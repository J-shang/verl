from verl.utils.reward_score import prime_math

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return prime_math.compute_score(solution_str, ground_truth)
