import torch
from dataclasses import dataclass
from functools import wraps


def divide_by_zero(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ZeroDivisionError:
            return torch.tensor(0.0)
    return wrapper


@dataclass
class ConfusionMatrix:
    tp: torch.Tensor = torch.tensor(0.0)
    tn: torch.Tensor = torch.tensor(0.0)
    fp: torch.Tensor = torch.tensor(0.0)
    fn: torch.Tensor = torch.tensor(0.0)

    @divide_by_zero
    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    @divide_by_zero
    def precision(self):
        return self.tp / (self.tp + self.fp)

    @divide_by_zero
    def recall(self):
        return self.tp / (self.tp + self.fn)

    @divide_by_zero
    def f1(self):
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())


def get_conf_matrix(labels: torch.Tensor, pred: torch.Tensor, num_classes: int) -> ConfusionMatrix:

    tp = torch.zeros(num_classes)
    tn = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)

    for i in range(num_classes):
        tp[i] = torch.sum((pred == i) & (labels == i)).item()
        tn[i] = torch.sum((pred != i) & (labels != i)).item()
        fp[i] = torch.sum((pred == i) & (labels != i)).item()
        fn[i] = torch.sum((pred != i) & (labels == i)).item()

    return ConfusionMatrix(tp, tn, fp, fn)
