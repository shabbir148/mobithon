import flask
from flask import Flask, jsonify, request
import cv2
import numpy as np
import io
import torch
from PIL import Image
import os
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.datasets as datasets
from PIL import Image  # Load img
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
app = Flask(__name__)


class FaceDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        first_img_tuple = random.choice(self.data.imgs)
        isSame = random.randint(0, 1)

        if isSame:
            while True:
                second_img_tuple = random.choice(self.data.imgs)
                if first_img_tuple[1] == second_img_tuple[1]:
                    break
        else:
            while True:
                second_img_tuple = random.choice(self.data.imgs)
                if first_img_tuple[1] != second_img_tuple[1]:
                    break

        img1 = Image.open(first_img_tuple[0]).convert('L')
        img2 = Image.open(second_img_tuple[0]).convert('L')
        label = torch.from_numpy(np.array([int(first_img_tuple[1] != second_img_tuple[1])], dtype=np.float32))

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    def __len__(self):
        return len(self.data.imgs)


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 2)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1,


def prepareImage(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    img = np.array(img)
    return img

@app.route('/predict', methods=['POST'])
def infer_image():
    # Catch the image file from a POST request
    print(request.files)
    file = flask.request.files.get('image', '')

    if not file:
        return

    # Read the image

    img_bytes = file.read()
    img = prepareImage(img_bytes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Siamese().to(device)
    net.load_state_dict("/model/model_300_epoch.pth")
    print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    folder_dataset_test = datasets.ImageFolder(root="data/data/faces/testing")
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])
    siamese_dataset = FaceDataset(folder_dataset_test, transform)
    test_dataloader = DataLoader(siamese_dataset, num_workers=2, batch_size=1, shuffle=True)

    # Grab one image that we are going to test
    dataiter = iter(test_dataloader)
    x0, name1 = gray    , "Tom"shabir
    x0 = torch.unsqueeze(transform(x0), 0)
    print(x0.shape)

    _min = 1000000
    lb = "not tom"
    name2 = 'not tom name'
    for i in range(37):
        # Iterate over 10 images and test them with the first image (x0)
        _, x1, label2 = next(dataiter)

        # Concatenate the two images together
        concatenated = torch.cat((x0, x1), 0)

        print(x0.shape, x1.shape)
        output1, output2 = net(x0.to(device), x1.to(device))

        euclidean_distance = F.pairwise_distance(output1, output2)

        if _min > euclidean_distance.item() and euclidean_distance.item() > 0.1:
            _min = euclidean_distance.item()
            lb = label2
        imshow(torchvision.utils.make_grid(concatenated),
               f'Dissimilarity: {euclidean_distance.item():.2f}, {name1, name2}')
    print(_min, lb, 'aaaaaa')

    # Return on a JSON format
    return "success"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')