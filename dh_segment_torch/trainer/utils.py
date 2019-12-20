import torch


def cut_with_padding(input_tensor, shape, margin=0):
    return input_tensor[..., margin:shape[0].item() - margin, margin:shape[1].item() - margin]


def compute_with_shapes(input_tensor, shapes, reduce=torch.mean, margin=0):
    res = torch.tensor(0.0).to(input_tensor.device)
    for idx in range(shapes.shape[0]):
        shape = shapes[idx]
        res += reduce(cut_with_padding(input_tensor[idx], shape, margin))  # .item()
    res = reduce(res)
    return res


def patch_loss_with_padding(loss_class, margin=0):
    class LossPatchedWithPadding(loss_class):
        def __init__(self, *args, **kwargs):
            kwargs['reduction'] = 'none'
            super().__init__(*args, **kwargs)

        def forward(self, input, target):
            target, shapes = target
            loss = super().forward(input, target)
            return compute_with_shapes(loss, shapes, margin=margin)

    return LossPatchedWithPadding


def patch_metric_with_padding(metric_class, margin=0):
    class MetricWithPadding(metric_class):
        def update(self, output):
            y_pred, (y, shapes) = output
            for idx in range(shapes.shape[0]):
                shape = shapes[idx]
                y_pred_tmp = cut_with_padding(y_pred[idx], shape, margin).unsqueeze(0).contiguous()
                y_tmp = cut_with_padding(y[idx], shape, margin).unsqueeze(0).contiguous()
                super().update((y_pred_tmp, y_tmp))

    return MetricWithPadding


def to_onehot(indices, num_classes):
    """Convert a tensor of indices of any shape `(N, ...)` to a
    tensor of one-hot indicators of shape `(N, num_classes, ...) and of type uint8. Output's device is equal to the
    input's device`.
    """
    onehot = torch.zeros(indices.shape[0], num_classes, *indices.shape[1:],
                         dtype=torch.float32,
                         device=indices.device)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)