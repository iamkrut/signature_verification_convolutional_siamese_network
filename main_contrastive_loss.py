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
    train_number_epochs = 20
    test_batch_size = 16
    lr = 1e-4
    file_prefix = "contrastive_"

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


def compute_accuracy_roc(predictions, labels, d=0.5):
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)

    idx1 = predictions.ravel() <= d
    idx2 = predictions.ravel() > d

    tp = np.sum(labels[idx1] == 1)
    fp = np.sum(labels[idx1] == 0)
    tn = np.sum(labels[idx2] == 0)
    fn = np.sum(labels[idx2] == 1)

    print(tp, fp, nsame)
    print(tn, fn, ndiff)
    acc = (tp + tn) / labels.shape[0]

    return acc


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
                self.data.extend([(obs[0], obs[1], 1.0) for obs in self.org[_id]])
                self.data.extend([(obs[0], obs[1], 0.0) for obs in self.forg[_id]])
        else:
            val_ids = list(set(self.org.keys()).difference(set(train_ids))) # getting the validation ids
            for _id in val_ids:
                self.data.extend([(obs[0], obs[1], 1.0) for obs in self.org[_id]])
                self.data.extend([(obs[0], obs[1], 0.0) for obs in self.forg[_id]])

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


class BHSig260Dataset(Dataset):

    def __init__(self, folder, dataset_pair_file, transform=None, should_invert=True):
        self.folder = folder
        self.dataset_pair_file = self.folder + dataset_pair_file
        self.transform = transform
        self.should_invert = should_invert
        self.data = []

        with open(self.dataset_pair_file) as fp:
            for _, line in enumerate(fp):
                pair = line.strip().split(" ")
                self.data.append((pair[0], pair[1], float(pair[2])))

    def __getitem__(self, index):

        img_0, img_1, label = self.data[index]

        img_0 = Image.open(self.folder + img_0)
        img_1 = Image.open(self.folder + img_1)
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
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      ((1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)))

        return loss_contrastive


if __name__ == '__main__':

    train = False
    is_cuda = torch.cuda.is_available()
    print("Is cuda availiable: ", is_cuda)

    if train:
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

        # example_batch = next(dataiter)
        # concatenated = torch.cat((example_batch[0],example_batch[1]),0)
        # # torchvision.utils.save_image((concatenated), 'train_data.png')
        # imshow(torchvision.utils.make_grid(concatenated))
        # print(example_batch[2].numpy())

        train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=Config.train_batch_size)
        val_dataloader = DataLoader(val_dataset, shuffle=True, num_workers=2, batch_size=Config.val_batch_size)

        net = SigNetModel()

        criterion = ContrastiveLoss()
        # optimizer = optim.RMSprop(net.parameters(), lr=Config.lr, eps=1e-8, weight_decay=0.0005, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=Config.lr, weight_decay=0.0001)

        if is_cuda:
            net.cuda()

        metric_history = defaultdict(list)

        print(net)
        print("Train data: ", len(train_dataset))
        print("Val data: ", len(val_dataset))
        past_acc = 0.0
        for epoch in range(0, Config.train_number_epochs):

            # adjusting learning rate
            # lr = Config.lr * (0.1 ** epoch)
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr
            #
            # print("lr: ", lr)

            acc_train_loss = 0
            acc_val_loss = 0
            avg_acc = 0
            avg_distance_threshold = 0
            val_predictions = []
            val_labels = []

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
            metric_history['train'].append(acc_train_loss / (i+1))

            # validation
            net.eval()
            for i, data in enumerate(val_dataloader, 0):
                img0, img1, label = data
                if is_cuda:
                    img0, img1 , label = img0.cuda(), img1.cuda(), label.cuda()
                output1, output2 = net(img0, img1)
                loss_contrastive = criterion(output1, output2, label)
                acc_val_loss += loss_contrastive.item()

                # predictions
                euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
                if is_cuda:
                    euclidean_distance = euclidean_distance.cpu()
                    label = label.cpu()
                val_predictions.extend(euclidean_distance.detach().numpy())
                val_labels.extend(label.detach().numpy())

                # concatenated = torch.cat((img0, img1), 0)
                # torchvision.utils.save_image((concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()), 'val_'+epoch+'.png')

            acc = compute_accuracy_roc(np.array(val_predictions), np.array(val_labels))
            if acc > past_acc:
                past_acc = acc
                torch.save(net.state_dict(), Config.file_prefix + 'cedar_cl.dth')
                print("Saving the model parameters")

            metric_history['val'].append(acc_val_loss / (i+1))
            metric_history['acc'].append(acc)
            print("Epoch number {} Train loss {} Val loss {}".format(epoch, metric_history['train'][-1], metric_history['val'][-1]))
            print('Val accuracy {}\n'.format(acc))

        # plot loss metric
        plt.figure()
        plt.clf()
        iter = [i for i in range(0, epoch+1)]
        plt.plot(iter, metric_history['train'], '-r', label='train loss')
        plt.plot(iter, metric_history['val'], '-b', label='val loss')
        # plt.plot(iter, metric_history['acc'], '-g', label='val accuracy')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend(loc='upper left')
        plt.savefig(Config.file_prefix+'loss_plot.png')

        # plot accuracy
        plt.figure()
        iter = [i for i in range(0, epoch + 1)]
        plt.plot(iter, metric_history['acc'], '-g', label='accuracy')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend(loc='upper left')
        plt.savefig(Config.file_prefix + 'acc_plot.png')

    else:
        print("Testing on BHSig260 Bengali dataset")
        # Testing on BHSig260
        test_dataset = BHSig260Dataset(folder="BHSig260/Bengali/", dataset_pair_file="Bengali_pairs.txt", transform=transforms.Compose(
                                                  [transforms.Resize((155, 220), interpolation=PIL.Image.BILINEAR),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.5], std=[0.5])]),
                                        should_invert=True)

        test_dataloader = DataLoader(test_dataset, shuffle=True, num_workers=8, batch_size=Config.test_batch_size)

        net = SigNetModel()

        # loading net state dict
        if is_cuda:
            net.load_state_dict(torch.load("contrastive_cedar_cl.dth"))
        else:
            net.load_state_dict(torch.load("contrastive_cedar_cl.dth", map_location=torch.device('cpu')))

        if is_cuda:
            net.cuda()

        print("Test data: ", len(test_dataset))

        test_predictions = []
        test_labels = []

        # testing
        net.eval()
        for i, data in enumerate(test_dataloader, 0):
            img0, img1, label = data
            if is_cuda:
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            output1, output2 = net(img0, img1)

            # predictions
            euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
            if is_cuda:
                euclidean_distance = euclidean_distance.cpu()
                label = label.cpu()
            test_predictions.extend(euclidean_distance.detach().numpy())
            test_labels.extend(label.detach().numpy())

            # concatenated = torch.cat((img0, img1), 0)
            # torchvision.utils.save_image((concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()), 'val_'+epoch+'.png')

        acc = compute_accuracy_roc(np.array(test_predictions), np.array(test_labels))

        print('Test accuracy {}\n'.format(acc))