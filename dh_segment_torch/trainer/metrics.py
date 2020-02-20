from abc import ABCMeta

import torch
from torch.optim.lr_scheduler import _LRScheduler


class Metric(metaclass=ABCMeta):
    def update(self, output):
        """

        :param output: (y_pred, y, loss)
        :return:
        """
        raise NotImplementedError("Should implement update class")

    def reset(self):
        pass

    @property
    def value(self):
        return 0


class LearningRate(Metric):
    def __init__(self, learning_rate_scheduler: _LRScheduler):
        self.learning_rate_scheduler = learning_rate_scheduler

    def update(self, output):
        pass

    @property
    def value(self):
        lr = self.learning_rate_scheduler.get_last_lr()
        if len(lr) == 1:
            lr = lr[0]
        return lr


class AverageLoss(Metric):
    def __init__(self):
        self.n_samples = torch.tensor(0, dtype=torch.double).cpu()
        self.loss_sum = torch.tensor(0, dtype=torch.double).cpu()

    def update(self, output):
        _, _, loss = output
        self.n_samples += 1.0
        self.loss_sum += loss

    def reset(self):
        self.n_samples = torch.tensor(0, dtype=torch.double).cpu()
        self.loss_sum = torch.tensor(0, dtype=torch.double).cpu()

    @property
    def value(self):
        if self.n_samples == 0:
            raise ValueError("Cannot compute average with 0 samples")
        return self.loss_sum / self.n_samples


class ConfusionMatrix(Metric):
    def __init__(self, n_classes, is_multilabel=False):
        self.n_classes = n_classes
        self.is_multilabel = is_multilabel
        if is_multilabel:
            self.confusion_matrix = torch.zeros(
                (n_classes, 2, 2), dtype=torch.int64
            ).cpu()
        else:
            self.confusion_matrix = torch.zeros(
                (n_classes, n_classes), dtype=torch.int64
            ).cpu()

    def update(self, output):
        y_pred, y, _ = output
        y_pred = y_pred.detach().cpu()
        y = y.detach().cpu()

        if self.is_multilabel:
            probas = torch.sigmoid(y_pred)

            for class_idx in range(self.n_classes):
                y_pred_class = probas[:, class_idx] > 0.5
                y_class = y[:, class_idx]
                cm = compute_cm(y_pred_class, y_class, 2)
                self.confusion_matrix[class_idx] += cm
        else:
            y_pred = torch.argmax(y_pred, dim=1)
            cm = compute_cm(y_pred, y, self.n_classes)
            self.confusion_matrix += cm

    def reset(self):
        if self.is_multilabel:
            self.confusion_matrix = torch.zeros(
                (self.n_classes, 2, 2), dtype=torch.int64
            ).cpu()
        else:
            self.confusion_matrix = torch.zeros(
                (self.n_classes, self.n_classes), dtype=torch.int64
            ).cpu()

    @property
    def value(self):
        return self.confusion_matrix


def compute_cm(y_pred, y, n_classes):
    y_pred = y_pred.flatten().long()
    y = y.flatten().long()

    cm = torch.bincount(y_pred + n_classes * y, minlength=n_classes ** 2)

    return cm.reshape((n_classes, n_classes))


class WrappingMetric(object):
    def __init__(self, metric):
        self.metric = metric

    def update(self, output):
        pass

    @property
    def value(self):
        metric = self.metric.value
        # If multilabel
        if len(metric.shape) > 2:
            n_classes = metric.shape[0]
            iou = torch.empty(n_classes)
            for class_idx in range(n_classes):
                # Compute metric as binary, get result of class (index 1)
                iou[class_idx] = self.compute_metric(metric[class_idx])[1]
            return iou
        else:
            return self.compute_metric(metric)

    def reset(self):
        self.metric.reset()

    @staticmethod
    def compute_metric(metric):
        return metric


class IoU(WrappingMetric):
    @staticmethod
    def compute_metric(cm):
        cm = cm.double()
        iou = cm.diag() / (cm.sum(dim=0) + cm.sum(dim=1) - cm.diag() + 1e-15)
        return iou


class Precision(WrappingMetric):
    @staticmethod
    def compute_metric(cm):
        cm = cm.double()
        precision = cm.diag() / (cm.sum(dim=1) + 1e-15)
        return precision


class Recall(WrappingMetric):
    @staticmethod
    def compute_metric(cm):
        cm = cm.double()
        recall = cm.diag() / (cm.sum(dim=0) + 1e-15)
        return recall


class Accuracy(WrappingMetric):
    @staticmethod
    def compute_metric(cm):
        cm = cm.double()
        accuracy = cm.diag() / (cm.sum(dim=0) + cm.sum(dim=1) + 1e-15)
        return accuracy


class mIoU(IoU):
    @property
    def value(self):
        return super().value.mean()


class AveragePrecision(Precision):
    @property
    def value(self):
        return super().value.mean()


class AverageRecall(Recall):
    @property
    def value(self):
        return super().value.mean()


class AverageAccuracy(Accuracy):
    @property
    def value(self):
        return super().value.mean()
