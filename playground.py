import torch
from dataclasses import dataclass

num_classes = 10


labels = [7, 6, 8, 8, 6, 4, 9, 1, 2, 5, 1, 2, 6, 3, 9, 0]
pred =   [5, 0, 0, 5, 0, 3, 3, 5, 0, 5, 5, 3, 3, 5, 0, 3]

l = torch.tensor(labels)
p = torch.tensor(pred)

tp = torch.zeros(10)
tn = torch.zeros(10)
fp = torch.zeros(10)
fn = torch.zeros(10)

for i in range(num_classes):
    tp[i] = torch.sum((p == i) & (l == i)).item()
    tn[i] = torch.sum((p != i) & (l != i)).item()
    fp[i] = torch.sum((p == i) & (l != i)).item()
    fn[i] = torch.sum((p != i) & (l == i)).item()

print(tp, tn, fp, fn)

# per class

conf_matrix = ConfMatrix(tp, tn, fp, fn)
print(conf_matrix.accuracy().mean())
print(conf_matrix.precision().mean())
print(conf_matrix.recall().mean())
print(conf_matrix.f1().mean())
