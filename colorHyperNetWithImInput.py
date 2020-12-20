import torch
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize, downscale_local_mean
from torch import nn
import joblib
from PIL import Image
import numpy as np
from matplotlib import cm
from matplotlib.patches import Rectangle
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import os
from random import shuffle

# This file attempts to train the hypernet to predict the color of the image.
# SimpleNet is g, HyperNet2 is f

EPOCHS = 10
BATCH_SIZE = 100
LEARNING_RATE = 1e-3
H = 32
W = 32

def make_part_of_data(part, size, in_path, out_path):
    in_ims = []
    cnt = 0
    for filename in tqdm(os.listdir(in_path)):
        full_path = os.path.join(in_path, filename)
        im = cv2.imread(full_path)
        im = im / 255.0
        in_ims += [im]
        if cnt > size * part:
            break
        cnt += 1
    print(len(in_ims))
    joblib.dump(in_ims, out_path)


def make_data(part):
    train_dir = r"D:\data\cifar-10\train"
    test_dir = r"D:\data\cifar-10\test"
    train_out_path = r"D:\data\cifar-10\train.pickle"
    test_out_path = r"D:\data\cifar-10\test.pickle"
    make_part_of_data(part, 50000, train_dir, train_out_path)
    make_part_of_data(part, 10000, test_dir, test_out_path)


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out

class HyperNet2(nn.Module):
    def __init__(self, out_size):
        super(HyperNet2, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(6, 20, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(20, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(32, out_size)
        )

    # Defining the forward pass
    def forward(self, x):
        ret_batch = torch.zeros([BATCH_SIZE, H, W, 3])
        x_proc = self.cnn_layers(x)
        x_proc = x_proc.view(x_proc.size(0), -1)
        pred_batch = self.linear_layers(x_proc)
        for k in range(BATCH_SIZE):
            pred = pred_batch[k, :]
            simple_net = SimpleNet(2*pred)
            locs_vec = torch.FloatTensor([[i / H, j / W, x[k, 0, i, j]] for i in range(H) for j in range(W)])
            colored = simple_net(locs_vec)
            """
            colored = torch.zeros([h, w, 3])
            for i in range(h):
                for j in range(w):
                    colored[i][j] = simple_net(torch.FloatTensor([i/h, j/w]))
            """
            ret_batch[k, :, :, :] = torch.reshape(colored, [H, W, 3])
        return ret_batch


def init_layer(layer, weights, offset):
    w_size = torch.numel(layer.weight)
    b_size = torch.numel(layer.bias)
    layer.weight.data = torch.reshape(weights[offset: offset+w_size], layer.weight.size())
    offset += w_size
    layer.bias.data = torch.reshape(weights[offset: offset+b_size], layer.bias.size())
    offset+=b_size
    return offset


class SimpleNet(nn.Module):
    def __init__(self, weights):
        super(SimpleNet, self).__init__()
        layers = [nn.Linear(3, 8), nn.Linear(8, 32), nn.Linear(32, 32), nn.Linear(32, 8), nn.Linear(8, 3)]
        offset = 0
        for layer in layers:
            offset = init_layer(layer, weights, offset)
        self.tot_size = offset
        self.model = nn.Sequential(layers[0], nn.ReLU(),
                                   layers[1], nn.ReLU(),
                                   layers[2], nn.ReLU(),
                                   layers[3], nn.ReLU(),
                                   layers[4], nn.Sigmoid())

    def forward(self, x):
        pred = self.model(x)
        return pred

    def predict(self, x):
        pred = self.model(x)
        return pred


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

#make_data(0.05)

train_ims = joblib.load(r"D:\data\cifar-10\train.pickle")
shuffle(train_ims)
test_ims = joblib.load(r"D:\data\cifar-10\test.pickle")
shuffle(test_ims)

#part = 0.1
#train_ims = train_ims[:part*len(train_ims)]
#train_ims = train_ims[:part*len(train_ims)]

#df = pd.DataFrame.from_records(df)
#df = df.sample(frac=1)
#df_train, df_test = train_test_split(df, train_size=0.8)


x = [0]*3000
for i in range(3000):
    x[i] = i
simple_net = SimpleNet(torch.FloatTensor(x))
net = HyperNet2(simple_net.tot_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), LEARNING_RATE)
loss_values_train = []
loss_values_test = []

for epoch in tqdm(range(EPOCHS)):
    shuffle(train_ims)
    curr_loss = 0
    num_batchs = len(train_ims)//BATCH_SIZE
    for i in tqdm(range(num_batchs)):
        y_batch = train_ims[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        x_batch = [rgb2gray(im) for im in y_batch]

        y_batch = torch.FloatTensor(y_batch)
        x_batch = torch.FloatTensor(x_batch)

        optimizer.zero_grad()
        outputs = net(torch.reshape(x_batch, [BATCH_SIZE, 1, H, W]))
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        curr_loss += loss
        if i == 0:
            plt.imshow(y_batch[0])
            plt.show()
            plt.imshow(outputs.detach()[0])
            plt.show()
        # print(loss)


    #loss_values_test += []
    loss_values_train += [curr_loss/num_batchs]
    print(loss_values_train[-1])

plot_loss = True
if plot_loss:
    p1 = plt.plot(np.array(loss_values_train), 'r', label="train")
    #p2 = plt.plot(np.array(loss_values_test), 'b', label="test")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("CE Loss")
    plt.title("Loss Over Time")
    plt.show()
