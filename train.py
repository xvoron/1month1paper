import csv
from dataclasses import dataclass
import logging
import os
from typing import Literal

from PIL import Image
import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from VGGNet import VGGNet
from alexnet import AlexNet
from metrics import ConfusionMatrix, get_conf_matrix


writer = SummaryWriter()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_PATH = 'datasets/10BigCats/WILDCATS.CSV'
DEFAULT_CHECKPOINT_PATH = './checkpoints'
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
EPOCHS = 500
DEFAULT_MODEL = 'alexnet'
NUM_CLASSES = 10

logger.info(f"Using {DEVICE} device")


def get_model(model: Literal["alexnet", "VGGNet"], num_classes: int):
    if model == 'alexnet':
        return AlexNet(num_classes)
    elif model == 'VGGNet':
        return VGGNet(num_classes)
    else:
        raise ValueError(f"Unknown model: {model}")



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
        image = Image.open(annotation.image_path)

        if self.augmentations:
            image = self.augmentations(image)
        else:
            image = transforms.ToTensor()(image)

        return image, torch.tensor(int(annotation.label))

    def __repr__(self):
        return f"Dataset with {len(self.annotations)} annotations."


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
                          annotation.mode == 'valid'], None)
    test_dataset = Data([annotation for annotation in annotations if
                         annotation.mode == 'test'], None)

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
    images, labels, pred = None, None, None
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    conf_matrix = ConfusionMatrix()

    with torch.no_grad():
        for images, labels in dataloader:

            images = images.to(DEVICE).float()
            labels = labels.to(DEVICE)

            pred = model(images)

            test_loss += loss_fn(pred, labels).item()
            pred = torch.argmax(pred, dim=1)

            conf_matrix = get_conf_matrix(pred, labels, num_classes=NUM_CLASSES)

    test_loss /= num_batches
    print(f"Epoch: {epoch} | Validation accuracy: {conf_matrix.accuracy().mean().item():.3f} | Avg loss: {test_loss:.3f}")
    writer.add_scalar('Validation loss', test_loss, epoch)
    writer.add_scalar('Validation accuracy', conf_matrix.accuracy().mean().item(), epoch)
    writer.add_scalar('Validation precision', conf_matrix.precision().mean().item(), epoch)
    writer.add_scalar('Validation recall', conf_matrix.recall().mean().item(), epoch)
    writer.add_scalar('Validation f1', conf_matrix.f1().mean().item(), epoch)

    # add images with labels to tensorboard
    if images is not None and labels is not None and pred is not None:
        for i, (image, label) in enumerate(zip(images, labels)):
            writer.add_image(f"Validation images/{i}", _classification_image_to_tensorboard(image, label, pred[i]), epoch)


def _classification_image_to_tensorboard(image, label, prediction):
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(f"Label: {label} | Prediction: {prediction}")
    fig.canvas.draw()

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image = image.transpose(2, 0, 1)
    plt.close(fig)

    return image


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str):
    os.makedirs(path, exist_ok=True)
    checkpoint_path = os.path.join(path, f'checkpoint_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)


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
@click.option('-fc', '--from_checkpoint', default=None, help='Path to the checkpoint to continue training from')
@click.option('-m', '--model', default=DEFAULT_MODEL, help='Model to use for training [alexnet, VGGNet]')
def main(path, checkpoint_path, from_checkpoint, model):
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

    model = get_model(model, NUM_CLASSES).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 0
    if from_checkpoint:
        start_epoch = load_checkpoint(from_checkpoint, model, optimizer)

    # epochs = tqdm(range(start_epoch, EPOCHS))
    epochs = range(start_epoch, EPOCHS)

    for epoch in epochs:
        train_loop(train_loader, model, loss_fn, optimizer, epoch)
        val_loop(val_loader, model, loss_fn, epoch)

        if epoch % 50 == 0 and epoch != 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_path)


if __name__ == '__main__':
    main()
