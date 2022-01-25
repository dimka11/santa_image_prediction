import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import os

from sklearn.model_selection import train_test_split

import time 
from tqdm import tqdm

import pandas as pd

def set_requires_grad(model, value=False):
    for param in model.parameters():
        param.requires_grad = value

def train_model(model, dataloaders, criterion, optimizer,
                phases, num_epochs=3):
    start_time = time.time()

    acc_history = {k: list() for k in phases}
    loss_history = {k: list() for k in phases}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            n_batches = len(dataloaders[phase])
            for inputs, labels in tqdm(dataloaders[phase], total=n_batches):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double()
            epoch_acc /= len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                       epoch_acc))
            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_acc)

        print()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                        time_elapsed % 60))

    return model, acc_history

def init_model(device):
    model = torchvision.models.mobilenet_v3_small(pretrained=True)
    set_requires_grad(model, False)
    model.classifier[3] = torch.nn.Linear(model.classifier[0].out_features, 3)
    model = model.to(device)
    return model


class DedyDataset(Dataset):
    def __init__(self, root_dir, csv_path=None, transform=None):
        
        self.transform = transform
        self.files = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]
        self.targets = None
        if csv_path:
            df = pd.read_csv(csv_path, sep="\t")
            self.targets = df["class_id"].tolist()
            self.files = [os.path.join(root_dir, fname) for fname in df["image_name"].tolist()]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        target = self.targets[idx] if self.targets else -1
        if self.transform:
            image = self.transform(image)
        return image, target

# hardcode
MODEL_WEIGHTS = ".data/weights/baseline.pt"
TRAIN_DATASET = "./data/train/"
TRAIN_CSV = "./data/train.csv"

if __name__ == "__main__":
    img_size = 224
    # make slight augmentation and normalization on ImageNet statistics
    trans = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dset = DedyDataset(TRAIN_DATASET, TRAIN_CSV, trans)
    labels = dset.targets
    indices = list(range(len(labels)))
    ind_train, ind_test, _, _ = train_test_split(indices, labels, test_size=0.2, random_state=139, stratify=labels)

    trainset = torch.utils.data.Subset(dset, ind_train)                           
    testset = torch.utils.data.Subset(dset, ind_test)

    batch_size = 16
    num_workers = 4
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    loaders = {'train': trainloader, 'val': testloader}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_model(device)

    pretrain_optimizer = torch.optim.SGD(model.classifier[3].parameters(),
                                        lr=0.001, momentum=0.9)
    train_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    # Pretrain
    # запустить предобучение модели на две эпохи
    pretrain_results = train_model(model, loaders, criterion, pretrain_optimizer,
                phases=['train', 'val'], num_epochs=5) # 1

    # Train
    # запустить дообучение модели
    set_requires_grad(model, True)
    train_results = train_model(model, loaders, criterion, train_optimizer,
                phases=['train', 'val'], num_epochs=25) # 0

    torch.save(model.state_dict(), MODEL_WEIGHTS)
