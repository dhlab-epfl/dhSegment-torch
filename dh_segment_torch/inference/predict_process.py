from typing import Dict, Optional, List

from torch.utils.data import DataLoader

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
    def __init__(
        self,
        data: InferenceDataset,
        model: InferenceModel,
        post_process: PostProcessingPipeline,
        batch_size: int = 1,
        num_workers: int = 0,
        index_to_name: Optional[Dict[int, str]] = None,
        add_path: bool = False,
        output_names: Optional[List[str]] = None,
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

        for example in data_loader:
            images_batch = example["input"]
            shapes = example["shapes"]
            paths = example["paths"]
            batch_probas = self.model.predict(images_batch, shapes).to("cpu").numpy()
            for probas, path in zip(batch_probas, paths):
                if self.index_to_name:
                    input_probas = {}
                    for index, name in self.index_to_name.items():
                        input_probas[name] = probas[index]
                    if self.add_path:
                        input_probas["path"] = path
                    complete_result = self.post_process.apply(**input_probas)
                else:
                    kwargs = {}
                    if self.add_path:
                        kwargs["path"] = path
                    complete_result = self.post_process.apply(probas, **kwargs)

                if self.output_names:
                    result = {}
                    for name in self.output_names:
                        result[name] = complete_result[name]
                else:
                    result = complete_result

                results.append(result)
        return results
