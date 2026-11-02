import numpy.typing as npt
import numpy as np
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = dataset.shape[0]

    start_index = np.random.randint(0, n - context_length, size=(batch_size, 1))
    offset = np.arange(0, context_length)
    inputs = torch.from_numpy(dataset[start_index + offset]).to(device, dtype=torch.long)
    targets = torch.from_numpy(dataset[start_index + offset + 1]).to(device, dtype=torch.long)

    return inputs, targets

def get_eval_batch(
    dataset: npt.NDArray, batch_idx: int, batch_size: int, context_length: int, device: str, window_stride: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return the batch_idx-th batch of non-overlapping windows for a pass over the dataset.

    Unlike get_batch (random starts, for training), windows start every context_length tokens.
    With window_stride=1 every token is a target exactly once across all batches (full pass);
    with window_stride=k only every k-th window is used, an evenly spaced fixed subset.
    The last batch may be smaller.

    Args:
        dataset: 1D array of token IDs.
        batch_idx: which batch to return, from 0 to ceil(num_windows / batch_size) - 1,
            where num_windows = ceil(((len(dataset) - 1) // context_length) / window_stride).
        batch_size: max number of windows per batch.
        context_length: tokens per window.
        device: device to place the tensors on.
        window_stride: use every window_stride-th window.

    Returns:
        inputs, targets: LongTensors of shape (<= batch_size, context_length),
            where targets are inputs shifted by one token.
    """
    n_tokens = dataset.shape[0]
    start_step = context_length * window_stride
    first_start = batch_idx * batch_size * start_step
    window_starts = np.arange(first_start, first_start + batch_size * start_step, start_step)
    # Keep windows whose last target (start + context_length) is still in the dataset
    window_starts = window_starts[window_starts <= n_tokens - context_length - 1].reshape(-1, 1)
    offset = np.arange(context_length)

    inputs = torch.from_numpy(dataset[window_starts + offset]).to(device, dtype=torch.long)
    targets = torch.from_numpy(dataset[window_starts + offset + 1]).to(device, dtype=torch.long)

    return inputs, targets