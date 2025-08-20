import torch
from verl.protocol import DataProto, DataProtoItem


def filter_by_mask(batch: DataProto, mask: torch.Tensor, num_trainer_replicas: int) -> DataProto:
    # Filter batch to keep only valid samples
    batch = batch[mask]
    # Round down to the nearest multiple of world size
    max_batch_size = (batch.batch['input_ids'].shape[0] // num_trainer_replicas) * num_trainer_replicas
    if not max_batch_size:
        # give up, you got everything either all wrong or right.
        return None

    size_mask = torch.zeros(batch.batch['input_ids'].shape[0], dtype=torch.bool)
    size_mask[:max_batch_size] = True
    batch = batch[size_mask]
    return batch