import os
from typing import Dict, Optional, List, Union

import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.data_loader import collate_fn
from dh_segment_torch.inference.inference_dataset import InferenceDataset
from dh_segment_torch.inference.inference_model import InferenceModel
from dh_segment_torch.post_processing.post_processing_pipeline import (
    PostProcessingPipeline,
)


def collate_fn_with_paths(examples):
    paths = [example["path"] for example in examples]
    collated = collate_fn(examples)
    collated["paths"] = paths

    return collated


class PredictProcess(Registrable):
    default_implementation = "default"

    def __init__(
        self,
        data: InferenceDataset,
        model: InferenceModel,
        post_process: Optional[PostProcessingPipeline] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        index_to_name: Optional[Dict[int, str]] = None,
        add_path: bool = False,
        output_names: Optional[Union[str, List[str]]] = None,
        progress: bool = True,
    ):
        self.data = data
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.post_process = post_process
        self.index_to_name = index_to_name
        self.add_path = add_path
        self.output_names = output_names
        self.progress = progress

    def process(self):
        results = []

        data_loader = DataLoader(
            self.data,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_with_paths,
        )

        for example in tqdm(data_loader):
            images_batch = example["input"]
            shapes = example["shapes"]
            paths = example["paths"]
            batch_probas = self.model.predict(images_batch, shapes).to("cpu").numpy()
            for probas, path in zip(batch_probas, paths):
                input_data = {}
                if self.index_to_name:
                    for index, name in self.index_to_name.items():
                        input_data[name] = probas[index]
                else:
                    input_data["probas"] = probas
                if self.add_path:
                    input_data["path"] = path

                if self.post_process:
                    result = self.post_process.apply(**input_data)
                else:
                    result = input_data

                if self.output_names:
                    if isinstance(self.output_names, str):
                        result = result[self.output_names]
                    else:
                        result_filtered = {}

                        for name in self.output_names:
                            result_filtered[name] = result[name]
                        result = result_filtered

                results.append(result)
        return results

    def process_to_probas_files(
        self, output_directory: str, prefix: str = "", suffix: str = ""
    ):
        data_loader = DataLoader(
            self.data,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_with_paths,
        )

        for example in tqdm(data_loader):
            images_batch = example["input"]
            shapes = example["shapes"]
            paths = example["paths"]
            batch_probas = self.model.predict(images_batch, shapes).to("cpu").numpy()
            for probas, path in zip(batch_probas, paths):
                basename = (
                    prefix
                    + os.path.splitext(os.path.basename(path))[0]
                    + suffix
                    + ".npy"
                )
                output_path = os.path.join(output_directory, basename)
                np.save(output_path, probas)


PredictProcess.register("default")(PredictProcess)
