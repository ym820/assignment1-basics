import numpy.typing as npt
import numpy as np
import torch


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = dataset.shape[0]

    start_index = np.random.randint(0, n - context_length, size=(batch_size, 1))
    offset = np.arange(0, context_length)
    inputs = torch.from_numpy(dataset[start_index + offset]).to(device, dtype=torch.long)
    targets = torch.from_numpy(dataset[start_index + offset + 1]).to(device, dtype=torch.long)

    return inputs, targets