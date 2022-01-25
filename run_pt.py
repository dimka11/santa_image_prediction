import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import pandas as pd
import os
import glob


class DedyDataset(Dataset):
    def __init__(self, root_dir, csv_path=None, transform=None):

        self.transform = transform
        self.files = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if not fname.startswith('.')]
        self.names = [fname for fname in os.listdir(root_dir) if not fname.startswith('.')]
        if csv_path:
            df = pd.read_csv(csv_path, sep="\t")
            self.names = df["class_id"].tolist()
            self.files = [os.path.join(root_dir, fname) for fname in df["image_name"].tolist()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        target = self.names[idx]
        if self.transform:
            image = self.transform(image)
        return image, target


if __name__ == "__main__":
    MODEL_WEIGHTS = "./data/weights/baseline.pt"
    TEST_DATASET = "./data/test/"
    img_size = 224

    # glob = glob.glob(f'{TEST_DATASET}/*.jpg')
    # [print(img) for img in glob]

    transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dset = DedyDataset(TEST_DATASET, transform=transforms)
    batch_size = 32
    test_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = torch.nn.Linear(model.classifier[0].out_features, 3)

    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model = model.to(device)
    print("model and weights loaded")

    batches_preds = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            names = batch[1]
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            print(names[0])
            batch_pred = pd.DataFrame({'image_name': names, 'class_id': preds.to("cpu")})
            batches_preds.append(batch_pred)

    submit = pd.concat(batches_preds)
    submit.to_csv('./data/out/submission.csv', sep='\t', index=False)

#%%
