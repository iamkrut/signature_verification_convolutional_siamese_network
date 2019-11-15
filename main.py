import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import re
from collections import defaultdict
from signet import SigNetModel
import itertools as it


def imshow(img,text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold', bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("grid.png")


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.savefig("loss.png")


class Config:
    original_dir = "./signatures/"
    forged_dir = "./signatures/full_forg/"
    train_batch_size = 16
    val_batch_size = 8
    train_number_epochs = 20


def format_image_name(filename):
    ref = re.findall(r'\d+', filename)
    return ref[0], ref[1]


def generate_permutations(list):
    permutations = []

    return permutations


def generate_permutations(list_1, list_2 = None):
    permutations = []

    if list_2 is not None:
        permutations.extend(list(it.product(list_1, list_2)))
    else:
        for a, b in it.combinations(list_1, 2):
            permutations.append((a, b))
    return permutations


class SiameseNetworkDataset(Dataset):

    def __init__(self, image_folder_dataset, transform=None, should_invert=True, no_train_signers=50, is_train=True):
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
        self.should_invert = should_invert
        self.org = defaultdict(list)
        self.forg = defaultdict(list)
        self.is_train = is_train

        for img_name, label in self.image_folder_dataset.imgs:
            _id, _ = format_image_name(img_name)
            self.org[int(_id)].append(img_name) if label is 1 else self.forg[int(_id)].append(img_name)

        # generating_permutations
        for _id in self.org.keys():
            self.forg[_id] = random.sample(generate_permutations(self.org[_id], self.forg[_id]), 276)
            self.org[_id] = generate_permutations(self.org[_id])

        # randomly sample signer _id's for training
        random.seed(3)
        train_ids = random.sample(list(self.org.keys()), no_train_signers)

        self.data = []

        # combine org and forg to form train or val data
        if self.is_train:
            for _id in train_ids:
                self.data.extend([(obs[0], obs[1], 0) for obs in self.org[_id]])
                self.data.extend([(obs[0], obs[1], 1) for obs in self.forg[_id]])
        else:
            val_ids = list(set(self.org.keys()).difference(set(train_ids))) # getting the validation ids
            for _id in val_ids:
                self.data.extend([(obs[0], obs[1], 0) for obs in self.org[_id]])
                self.data.extend([(obs[0], obs[1], 1) for obs in self.forg[_id]])

        self.org.clear()
        self.forg.clear()

    def __getitem__(self, index):

        img_0, img_1, label = self.data[index]

        img_0 = Image.open(img_0)
        img_1 = Image.open(img_1)
        img_0 = img_0.convert("L")
        img_1 = img_1.convert("L")

        if self.should_invert:
            img_0 = PIL.ImageOps.invert(img_0)
            img_1 = PIL.ImageOps.invert(img_1)

        if self.transform is not None:
            img_0 = self.transform(img_0)
            img_1 = self.transform(img_1)

        return img_0, img_1, label

    def __len__(self):
        return len(self.data)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


if __name__ == '__main__':

    is_cuda = torch.cuda.is_available()
    print("Is cuda availiable: ", is_cuda)

    folder_dataset = dset.ImageFolder(root=Config.original_dir)
    train_dataset = SiameseNetworkDataset(image_folder_dataset=folder_dataset,
                                          transform=transforms.Compose([transforms.Resize((155, 220)),
                                                                        transforms.ToTensor()]),
                                          should_invert=True,
                                          is_train=True)
    val_dataset = SiameseNetworkDataset(image_folder_dataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((155, 220)),
                                                                      transforms.ToTensor()]),
                                        should_invert=True,
                                        is_train=False)

    # # visual some sample data
    # vis_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=8)
    # dataiter = iter(vis_dataloader)
    #
    # example_batch = next(dataiter)
    # concatenated = torch.cat((example_batch[0],example_batch[1]),0)
    # imshow(torchvision.utils.make_grid(concatenated))
    # # print(example_batch[2].numpy())

    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=2, batch_size=Config.train_batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=1, batch_size=Config.val_batch_size)

    net = SigNetModel()

    criterion = ContrastiveLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=0.0004, eps=1e-08, weight_decay=0.0005, momentum=0.9)

    if is_cuda:
        net.cuda()

    loss_history = defaultdict(list)

    for epoch in range(0,Config.train_number_epochs):

        acc_train_loss = 0
        acc_val_loss = 0

        # training
        net.train()
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            if is_cuda:
                img0, img1 , label = img0.cuda(), img1.cuda(), label.cuda()
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            acc_train_loss += loss_contrastive.item()
        loss_history['train'].append(acc_train_loss / (i+1))

        # validation
        net.eval()
        for i, data in enumerate(val_dataloader, 0):
            img0, img1, label = data
            if is_cuda:
                img0, img1 , label = img0.cuda(), img1.cuda(), label.cuda()
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            acc_val_loss += loss_contrastive.item()
        loss_history['val'].append(acc_val_loss / (i+1))

        print("Epoch number {} Train loss {} Val loss {}".format(epoch, loss_history['train'][-1], loss_history['val'][-1]))

    torch.save(net.state_dict(), 'cedar_cl.dth')
    # show_plot(counter, loss_history)