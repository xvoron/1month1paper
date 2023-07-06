"""
Implementation of the alexnet paper:
- https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

And the training loop for 10 Big Cats of the Wild dataset:
- https://www.kaggle.com/datasets/gpiosenka/cats-in-the-wild-image-classification?resource=download

Using torch.
"""
import csv
from dataclasses import dataclass
import logging
import os
from typing import Literal

import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import cv2

writer = SummaryWriter()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_PATH = './10BigCats/WILDCATS.CSV'
DEFAULT_CHECKPOINT_PATH = './checkpoints'
BATCH_SIZE = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCHS = 30

logger.info(f"Using {DEVICE} device")


@dataclass
class Annotation:
    image_path: str
    label: str
    name: str
    mode: Literal['train', 'valid', 'test']


class Data(Dataset):
    def __init__(self, annotations, augmentations=None):
        self.annotations = annotations
        self.augmentations = augmentations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        annotation = self.annotations[item]
        image = cv2.imread(annotation.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            image = self.augmentations(image)
        else:
            image = transforms.ToTensor()(image)

        return image, torch.tensor(int(annotation.label))

    def __repr__(self):
        return f"Dataset with {len(self.annotations)} annotations."


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)        
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)

        self.flattern = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )
        
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.conv5(x)
        x = nn.functional.relu(x)
        x = self.max_pool(x)

        x = self.flattern(x)
        x = self.fc(x)
        return x



def get_model():
    return AlexNet()


def get_data(path_to_csv: str):
    with open(path_to_csv, 'r') as f:
        reader = csv.reader(f)
        return list(reader)


def get_annotations(data: list, path_to_dir: str):
    annotations = []
    for row in data[1:]:
        label, image_path, name, mode, *_ = row
        annotations.append(Annotation(os.path.join(path_to_dir, image_path), label, name, mode))
    return annotations

def get_datasets(annotations, augmentations):
    train_dataset = Data([annotation for annotation in annotations if
                          annotation.mode == 'train'], augmentations)
    valid_dataset = Data([annotation for annotation in annotations if
                          annotation.mode == 'valid'], augmentations)
    test_dataset = Data([annotation for annotation in annotations if
                         annotation.mode == 'test'], augmentations)

    return train_dataset, valid_dataset, test_dataset

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    loss_data = 0
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        pred = model(images)
        loss = loss_fn(pred, labels)
        loss_data += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.add_scalar('Training loss', loss_data/size, epoch)


def val_loop(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    tp, tn, fp, fn = 0, 0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE).float()
            labels = labels.to(DEVICE)

            pred = model(images)
            test_loss += loss_fn(pred, labels).item()
            pred = torch.argmax(pred, dim=1)
            tp += torch.sum((pred == 1) & (labels == 1)).item()
            tn += torch.sum((pred == 0) & (labels == 0)).item()
            fp += torch.sum((pred == 1) & (labels == 0)).item()
            fn += torch.sum((pred == 0) & (labels == 1)).item()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    test_loss /= num_batches
    print(f"Test Error: \n Accuracy: {accuracy}; Avg loss: {test_loss:.3f}\n")
    writer.add_scalar('Validation loss', test_loss, epoch)
    writer.add_scalar('Validation accuracy', accuracy, epoch)
    writer.add_scalar('Validation precision', precision, epoch)
    writer.add_scalar('Validation recall', recall, epoch)
    writer.add_scalar('Validation f1', f1, epoch)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer):
    if not os.path.isfile(path):
        return 0

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


@click.command()
@click.option('-p', '--path', default=DEFAULT_PATH, help='Path to the dataset')
@click.option('-cp', '--checkpoint_path', default=DEFAULT_CHECKPOINT_PATH, help='Path to the checkpoint')
def main(path, checkpoint_path):
    data = get_data(path)
    annotations = get_annotations(data, path_to_dir=os.path.dirname(path))

    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
    ])
    train, val, _ = get_datasets(annotations, augmentations)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)

    model = get_model().to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
    for epoch in tqdm(range(start_epoch, EPOCHS)):
        train_loop(train_loader, model, loss_fn, optimizer, epoch)
        val_loop(val_loader, model, loss_fn, epoch)
        save_checkpoint(model, optimizer, epoch, checkpoint_path)


if __name__ == '__main__':
    main()
