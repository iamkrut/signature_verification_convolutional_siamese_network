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
from tripletnet import TripletNetModel
import itertools as it


def imshow(img,text=None, should_save=False):
    plt.figure()
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold', bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(Config.file_prefix+"grid.png")


# def show_plot(iteration, loss):
#     plt.plot(iteration, loss)
#     plt.savefig("loss.png")


class Config:
    original_dir = "./signatures/"
    forged_dir = "./signatures/full_forg/"
    train_batch_size = 16
    val_batch_size = 8
    train_number_epochs = 50
    lr = 1e-3
    file_prefix = "triplet_"


def format_image_name(filename):
    ref = re.findall(r'\d+', filename)
    return ref[0], ref[1]


def generate_permutations(list_1, list_2=None):
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

            for i, org_tuple in enumerate(self.org[_id]):
                o1, o2 = org_tuple
                self.org[_id][i] = (o1, o2, self.forg[_id][i][1])

        # randomly sample signer _id's for training
        random.seed(3)
        train_ids = random.sample(list(self.org.keys()), no_train_signers)

        self.data = []

        # combine org and forg to form train or val data
        if self.is_train:
            for _id in train_ids:
                self.data.extend([(obs[0], obs[1], obs[2]) for obs in self.org[_id]])
        else:
            val_ids = list(set(self.org.keys()).difference(set(train_ids))) # getting the validation ids
            for _id in val_ids:
                self.data.extend([(obs[0], obs[1], obs[2]) for obs in self.org[_id]])

        self.org.clear()
        self.forg.clear()

    def __getitem__(self, index):

        img_0, img_1, img_2 = self.data[index]

        img_0 = Image.open(img_0)
        img_1 = Image.open(img_1)
        img_2 = Image.open(img_2)
        img_0 = img_0.convert("L")
        img_1 = img_1.convert("L")
        img_2 = img_2.convert("L")

        if self.should_invert:
            img_0 = PIL.ImageOps.invert(img_0)
            img_1 = PIL.ImageOps.invert(img_1)
            img_2 = PIL.ImageOps.invert(img_2)

        if self.transform is not None:
            img_0 = self.transform(img_0)
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return img_0, img_1, img_2

    def __len__(self):
        return len(self.data)


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.accuracy = []
        self.margin_loss = torch.nn.MarginRankingLoss(margin=self.margin)

    def forward(self, distance_positive, distance_negative, target, size_average=True, is_val=False):
        losses = self.margin_loss(distance_positive, distance_negative, target)

        if is_val:
            pred = (distance_positive - distance_negative - self.margin).data
            self.accuracy.append(((pred > 0).sum()*1.0/distance_positive.size()[0]).item())

        return losses.mean() if size_average else losses.sum()


if __name__ == '__main__':

    is_cuda = torch.cuda.is_available()
    print("Is cuda availiable: ", is_cuda)

    folder_dataset = dset.ImageFolder(root=Config.original_dir)
    train_dataset = SiameseNetworkDataset(image_folder_dataset=folder_dataset,
                                          transform=transforms.Compose([transforms.Resize((155, 220), interpolation=PIL.Image.BILINEAR),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize(mean=[0.5], std=[0.5])]),
                                          should_invert=True,
                                          is_train=True)
    val_dataset = SiameseNetworkDataset(image_folder_dataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((155, 220), interpolation=PIL.Image.BILINEAR),
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize(mean=[0.5], std=[0.5])]),
                                        should_invert=True,
                                        is_train=False)

    # visual some sample data
    vis_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=8)
    dataiter = iter(vis_dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0],example_batch[1]),0)
    # torchvision.utils.save_image((concatenated), 'train_data.png')
    # imshow(torchvision.utils.make_grid(concatenated))
    # print(example_batch[2].numpy())

    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=Config.train_batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=True, num_workers=2, batch_size=Config.val_batch_size)

    net = TripletNetModel()

    criterion = TripletLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=Config.lr, eps=1e-8, weight_decay=0.0005, momentum=0.9)

    if is_cuda:
        net.cuda()

    metric_history = defaultdict(list)

    print(net)
    print("Train data: ", len(train_dataset))
    print("Val data: ", len(val_dataset))

    for epoch in range(0, Config.train_number_epochs):

        acc_train_loss = 0
        acc_val_loss = 0

        # training
        net.train()
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, img2 = data
            if is_cuda:
                img0, img1 , img2 = img0.cuda(), img1.cuda(), img2.cuda()
            optimizer.zero_grad()
            output1, output2, output3 = net(img0, img1, img2)
            distance_positive = (output1 - output2).pow(2).sum(1)
            distance_negative = (output1 - output3).pow(2).sum(1)
            target = torch.FloatTensor(distance_positive.size()).fill_(1)
            if is_cuda:
                target = target.cuda()
            loss_triplet = criterion(distance_positive, distance_negative, target)
            loss_triplet.backward()
            optimizer.step()
            acc_train_loss += loss_triplet.item()
        metric_history['train'].append(acc_train_loss / (i+1))

        # validation
        net.eval()
        criterion.accuracy = []
        for i, data in enumerate(val_dataloader, 0):
            img0, img1, img2 = data
            if is_cuda:
                img0, img1 , img2 = img0.cuda(), img1.cuda(), img2.cuda()
            output1, output2, output3 = net(img0, img1, img2)

            distance_positive = (output1 - output2).pow(2).sum(1)
            distance_negative = (output1 - output3).pow(2).sum(1)
            target = torch.FloatTensor(distance_positive.size()).fill_(1)
            if is_cuda:
                target = target.cuda()
            loss_triplet = criterion(distance_positive, distance_negative, target, is_val=True)
            acc_val_loss += loss_triplet.item()

        acc = sum(criterion.accuracy) / len(criterion.accuracy)
        metric_history['val'].append(acc_val_loss / (i+1))
        # metric_history['acc'].append(acc)
        print("\nEpoch number {} Train loss {} Val loss {}".format(epoch, metric_history['train'][-1], metric_history['val'][-1]))
        print('Accuracy {}'.format(acc))

    torch.save(net.state_dict(), Config.file_prefix+'cedar_cl.dth')

    # plot loss metric
    # plt.figure()
    # plt.clf()
    # iter = [i for i in range(0, epoch+1)]
    # plt.plot(iter, metric_history['train'], '-r', label='train loss')
    # plt.plot(iter, metric_history['val'], '-b', label='val loss')
    # # plt.plot(iter, metric_history['acc'], '-g', label='val accuracy')
    # plt.xlabel("epochs")
    # plt.ylabel("loss")
    # plt.legend(loc='upper left')
    # plt.savefig('loss_plot.png')