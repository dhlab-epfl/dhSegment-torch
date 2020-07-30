from typing import List, Optional, Union

import numpy as np
import torch
from torchvision.models import vgg19


class TopologyLoss(torch.nn.Module):
    def __init__(
        self,
        layers_sel: Union[int, List[int]],
        labels_sel: Optional[List[int]] = None,
        multilabel: bool = False,
    ):
        super().__init__()
        features_extractor = vgg19(pretrained=True).features
        layers_indices = np.where(
            [isinstance(x, torch.nn.MaxPool2d) for x in features_extractor]
        )[0]
        self.features_extractors = []
        if isinstance(layers_sel, int):
            layers_sel = [layers_sel]
        for layer_sel in layers_sel:
            if 0 > layer_sel - 1 >= len(layers_indices):
                raise ValueError(
                    f"Cannot find layer for layer {layer_sel}, out of range for {len(layers_indices)}."
                )
            self.features_extractors.append(
                features_extractor[: layers_indices[layer_sel - 1]]
            )
        self.labels_sel = labels_sel
        self.multilabel = multilabel

        self.mse = torch.nn.MSELoss(reduction="mean")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.multilabel:
            probas = torch.sigmoid(logits)
        else:
            probas = torch.softmax(logits, dim=1)

        loss = torch.tensor(0.0).to(logits.device)

        if self.labels_sel:
            labels_sel = self.labels_sel
        else:
            labels_sel = list(range(logits.shape[1]))

        for label_index in labels_sel:
            for features_extractor in self.features_extractors:
                pred_features = features_extractor.to(logits.device)(
                    probas[:, label_index].unsqueeze(1).repeat(1, 3, 1, 1)
                )
                if self.multilabel:
                    labels = target[:, label_index]
                else:
                    labels = (target == label_index).to(torch.float)
                gt_features = features_extractor.to(logits.device)(
                    labels.unsqueeze(1).repeat(1, 3, 1, 1)
                )
                loss += self.mse(pred_features, gt_features)
        return loss
