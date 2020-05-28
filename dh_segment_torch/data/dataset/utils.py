import numpy as np
import torch
import torch.nn.functional as F


# TODO paddings center ?
def compute_paddings(heights, widths):
    max_height = np.max(heights)
    max_width = np.max(widths)

    paddings_height = max_height - heights
    paddings_width = max_width - widths
    paddings_zeros = np.zeros(len(heights), dtype=int)

    paddings = np.stack(
        [paddings_zeros, paddings_width, paddings_zeros, paddings_height]
    ).T
    return list(map(tuple, paddings))


def collate_fn(examples):
    if not isinstance(examples, list):
        examples = [examples]
    heights = np.array([x["shape"][0] for x in examples])
    widths = np.array([x["shape"][1] for x in examples])
    paddings = compute_paddings(heights, widths)
    images = []
    masks = []
    shapes_out = []
    for example, padding in zip(examples, paddings):

        image, label, shape = example["image"], example["label"], example["shape"]
        images.append(F.pad(image, padding))
        masks.append(F.pad(label, padding))
        shapes_out.append(shape)

    return {
        "images": torch.stack(images, dim=0),
        "labels": torch.stack(masks, dim=0),
        "shapes": torch.stack(shapes_out, dim=0),
    }

def patches_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset_size = len(dataset.dataframe)
    num_workers = worker_info.num_workers
    worker_id = worker_info.id

    if (
        dataset_size > num_workers
        and worker_id >= num_workers - dataset_size % num_workers
    ):
        offset = 1
    else:
        offset = 0

    items_per_worker = max(1, dataset_size // num_workers)
    start = (worker_id * items_per_worker + offset) % (dataset_size)
    end = (start + items_per_worker + offset) % (dataset_size + 1)
    dataset.dataframe = dataset.dataframe.iloc[start:end]