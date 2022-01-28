import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import math
import numpy as np
from torch.autograd import Variable
import random
import torch.optim as optim



# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Batch size during training
batch_size = 64
# Number of channels in the training images. For color images this is 3
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 20
# Number of training epochs
num_epochs = 10
# Learning rate for optimizers
lr = 0.1
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.1)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.layer1 = nn.ConvTranspose2d(nz, nz//2, 4, 2, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(nz//2,affine=True)
        self.layer2 = nn.ConvTranspose2d(nz//2, nz//4, 4, 2, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(nz // 4,affine=True)
        self.layer4 = nn.ConvTranspose2d(nz//4,1, 4, 2, 1, bias=False)
        self.norm3 = nn.BatchNorm2d(1,affine=True)

    def forward(self, input):
        out = self.layer1(input)
        out = self.norm1(out)
        out = F.leaky_relu(out,0.2, inplace=True)
        out = self.layer2(out)
        out = self.norm2(out)
        out = F.leaky_relu(out,0.2, inplace=True)
        out = self.layer4(out)
        out = self.norm3(out)
        out = torch.sigmoid(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.layer1 = nn.Conv2d(1, nz//4, 4, 2, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(nz//4, affine=True)
        self.layer2 = nn.Conv2d(nz//4, nz//2, 4, 2, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(nz//2, affine=True)
        self.layer3 = nn.Conv2d(nz//2, nz, 4, 2, 1, bias=False)
        self.norm3 = nn.BatchNorm2d(nz, affine=True)
        self.layer4 = nn.Linear(nz*6*6,1)

    def forward(self, input):
        out = self.layer1(input)
        out = self.norm1(out)
        out = F.leaky_relu(out,0.2, inplace=True)
        out = self.layer2(out)
        out = self.norm2(out)
        out = F.leaky_relu(out,0.2, inplace=True)
        out = self.layer3(out)
        out = self.norm3(out)
        out = F.leaky_relu(out, 0.2, inplace=True)
        out = out.flatten(start_dim=1)
        out = self.layer4(out)
        out = torch.sigmoid(out)
        return out


# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Initialize BCELoss function
criterion = nn.BCELoss()


# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.SGD(netD.parameters(), lr=lr)
optimizerG = optim.SGD(netG.parameters(), lr=lr)

def train(net,data_loader,epoch,optimizer,criterion):
    net.train()
    for i in range(epoch):
        for batch_id, (x_train, y_train) in enumerate(data_loader):
            if x_train.shape[0] < 64:
                continue
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            output = net(x_train)
            y_train = y_train.reshape(-1,1)   # MSELoss
            loss = criterion(output, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_id % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, batch_id * len(x_train), len(data_loader.dataset), 100. * batch_id / len(train_loader), loss.item()))

def get_normal_data(model_test, test_data,threshold):
    print("get_normal_data")
    normal_image = []
    normal_image_label = []
    test_output = model_test(test_data).detach().numpy().tolist()
    test_data = test_data.detach().numpy().tolist()
    for i in range(len(test_output)):
        if test_output[i][0] > threshold:
            normal_image.append(test_data[i])
            normal_image_label.append(1)
    features = torch.tensor(normal_image, dtype=torch.float32)
    labels = torch.tensor(normal_image_label, dtype=torch.float32)
    print('normal data num: ',features.shape)
    normal_dataset = TensorDataset(features,labels)
    return normal_dataset

def distribution_calculate(model_test,test_data,test_label):
    normal_total = 0
    abnormal_total = 0
    test_output = model_test(test_data).detach().numpy().tolist()
    Y_test = test_label.numpy().tolist()
    x_normal = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    x_abnormal = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    y_normal = [0,0,0,0,0,0,0,0,0,0,0]
    y_abnormal = [0,0,0,0,0,0,0,0,0,0,0]
    for i in range(len(Y_test)):
        if Y_test[i] == 0:
            abnormal_total += 1
            y_abnormal[math.floor(test_output[i][0]*10)] += 1
        else:
            normal_total += 1
            y_normal[math.floor(test_output[i][0]*10)] += 1
    print('normal: ',y_normal)
    print('abnormal: ',y_abnormal)
    plt.scatter(x_normal,y_normal,c='k')
    plt.scatter(x_abnormal,y_abnormal,c='r')
    plt.title('total number')
    plt.show()


def test_p(model_test, test_data, test_label,threshold):
    test_output = model_test(test_data).detach().numpy().tolist()
    for i in range(len(test_output)):
        if test_output[i][0] <= threshold:
            test_output[i][0] = 0
        else:
            test_output[i][0] = 1
    Y_test = test_label.numpy().tolist()
    cm = confusion_matrix(Y_test, test_output)
    print(classification_report(Y_test, test_output))
    f,ax=plt.subplots(figsize=(2,2))
    sns.heatmap(cm,annot=True,linewidth=0.1,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    plt.show()

if __name__ =='__main__':
    can_image = np.load("../data/dcgan/gear_image_data.npy")
    can_image_label = np.load('../data/dcgan/gear_image_label.npy')
    features = torch.tensor(can_image, dtype=torch.float32)

    features = features.reshape(-1,1,48,48)
    labels = torch.tensor(can_image_label, dtype=torch.float32)  # MSELoss -> torch.float32
    train_data, test_data, train_label, test_label = train_test_split(features, labels, random_state=1, train_size=0.7,
                                                                      test_size=0.3, stratify=labels)
    train_ids = TensorDataset(train_data, train_label)
    train_loader = DataLoader(dataset=train_ids, batch_size=batch_size, shuffle=True)

    ###################################################################################first discriminator train & get normal data for GAN trainning
    train(netD,train_loader,10,optimizerD,criterion)
    torch.save(netD, '../model/GIDS_DCGAN_KnownAttack_gear.pkl')
    netD = torch.load('../model/GIDS_DCGAN_KnownAttack_gear.pkl')
    netD.to('cpu')
    normal_dataset = get_normal_data(netD,features,0.5)
###################################################################################test for first discriminator
    print('result of the first discriminator for gear attack')
    print('DoS attack')    
    can_image = np.load("../data/dcgan/DoS_image_data.npy")
    can_image_label = np.load('../data/dcgan/DoS_image_label.npy')
    features = torch.tensor(can_image, dtype=torch.float32)
    features = features.reshape(-1, 1, 48, 48)
    labels = torch.tensor(can_image_label, dtype=torch.float32)  #MSELoss -> torch.float32
    distribution_calculate(netD, features, labels)
    test_p(netD, features, labels, 0.5)
    print('Fuzzy attack')
    can_image = np.load("../data/dcgan/Fuzzy_image_data.npy")
    can_image_label = np.load('../data/dcgan/Fuzzy_image_label.npy')
    features = torch.tensor(can_image, dtype=torch.float32)
    features = features.reshape(-1, 1, 48, 48)
    labels = torch.tensor(can_image_label, dtype=torch.float32)  #MSELoss -> torch.float32
    distribution_calculate(netD, features, labels)
    test_p(netD, features, labels, 0.5)
    print('gear attack')
    can_image = np.load("../data/dcgan/gear_image_data.npy")
    can_image_label = np.load('../data/dcgan/gear_image_label.npy')
    features = torch.tensor(can_image, dtype=torch.float32)
    features = features.reshape(-1, 1, 48, 48)
    labels = torch.tensor(can_image_label, dtype=torch.float32)  #MSELoss -> torch.float32
    distribution_calculate(netD, features, labels)
    test_p(netD, features, labels, 0.5)
    print('RPM attack')
    can_image = np.load("../data/dcgan/RPM_image_data.npy")
    can_image_label = np.load('../data/dcgan/RPM_image_label.npy')
    features = torch.tensor(can_image, dtype=torch.float32)
    features = features.reshape(-1, 1, 48, 48)
    labels = torch.tensor(can_image_label, dtype=torch.float32)  #MSELoss -> torch.float32
    distribution_calculate(netD, features, labels)
    test_p(netD, features, labels, 0.5)
# #################################################################
# ############################################gan trainning
    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    netD.apply(weights_init)
    optimizerD = optim.SGD(netD.parameters(), lr=lr)
    optimizerG = optim.SGD(netG.parameters(), lr=lr)
    
    normal_dataloader = DataLoader(dataset=normal_dataset, batch_size=64, shuffle=True)
    real_img = Variable(torch.ones(64, dtype=float).to(device), requires_grad=False)
    fake_img = Variable(torch.zeros(64, dtype=float).to(device), requires_grad=False)
    training_flag = 0
    g_max = 0
    for i in range(num_epochs):
        G_ideas = Variable(torch.Tensor(np.random.normal(0, 1, (64, nz, 6, 6))).cuda(), requires_grad=False)
        for batch_id, (x_train, y_train) in enumerate(normal_dataloader):
            if training_flag == 0:
                print('epoch = ', i, 'train for D')
                if len(x_train) < 64:
                    continue
                x_train = x_train.to(device)
                G_paintings = netG(G_ideas).to(device).detach()  # fake painting from G (random ideas)
                prob_artist0 = netD(x_train)  # D try to increase this prob
                prob_artist1 = netD(G_paintings)  # D try to reduce this prob
                print('batch_id',batch_id)
                print('train for D, prob_artist0: ', prob_artist0.mean())
                print('train for D,prob_artist1: ', prob_artist1.mean())
                if(prob_artist1.mean() < 0.5) and (prob_artist0.mean()>0.5):
                    training_flag =1
                real_loss = criterion(prob_artist0.to(float).reshape(64), real_img)
                fake_loss = criterion(prob_artist1.to(float).reshape(64), fake_img)
                D_loss = real_loss +fake_loss
                optimizerD.zero_grad()
                D_loss.backward()
                optimizerD.step()
            while(1):
                print('epoch = ', i, 'train for G')
                if len(x_train) < 64:
                    break
                G_paintings = netG(G_ideas).to(device)  # fake painting from G (random ideas)
                prob_artist1 = netD(G_paintings)
                print('batch_id: ', batch_id,' train for G,prob_artist1: ', prob_artist1.mean())
                if(prob_artist1.mean() >0.5):
                    training_flag =0
                    break
                g_max = g_max +1
                if g_max>300:
                    training_flag = 0
                    g_max=0
                    break
                G_loss = criterion(prob_artist1.to(float).reshape(64), real_img)
                optimizerG.zero_grad()
                G_loss.backward()
                optimizerG.step()
    torch.save(netD, '../model/DCGAN_D.pkl')
    torch.save(netG, '../model/DCGAN_G.pkl')
##############################################################################
###################################################################################test for second discriminator
    print('result of the second discriminator for unknown attacks')
    second_D = torch.load('../model/DCGAN_D.pkl')
    second_D.to('cpu')
    print('DoS attack')
    can_image = np.load("../data/dcgan/DoS_image_data.npy")
    can_image_label = np.load('../data/dcgan/DoS_image_label.npy')
    features = torch.tensor(can_image, dtype=torch.float32)
    features = features.reshape(-1, 1, 48, 48)
    labels = torch.tensor(can_image_label, dtype=torch.float32)  #MSELoss -> torch.float32
    distribution_calculate(second_D, features, labels)
    test_p(second_D, features, labels, 0.5)
    print('Fuzzy attack')
    can_image = np.load("../data/dcgan/Fuzzy_image_data.npy")
    can_image_label = np.load('../data/dcgan/Fuzzy_image_label.npy')
    features = torch.tensor(can_image, dtype=torch.float32)
    features = features.reshape(-1, 1, 48, 48)
    labels = torch.tensor(can_image_label, dtype=torch.float32)  #MSELoss -> torch.float32
    distribution_calculate(second_D, features, labels)
    test_p(second_D, features, labels, 0.5)
    print('gear attack')
    can_image = np.load("../data/dcgan/gear_image_data.npy")
    can_image_label = np.load('../data/dcgan/gear_image_label.npy')
    features = torch.tensor(can_image, dtype=torch.float32)
    features = features.reshape(-1, 1, 48, 48)
    labels = torch.tensor(can_image_label, dtype=torch.float32)  #MSELoss -> torch.float32
    distribution_calculate(second_D, features, labels)
    test_p(second_D, features, labels, 0.5)
    print('RPM attack')
    can_image = np.load("../data/dcgan/RPM_image_data.npy")
    can_image_label = np.load('../data/dcgan/RPM_image_label.npy')
    features = torch.tensor(can_image, dtype=torch.float32)
    features = features.reshape(-1, 1, 48, 48)
    labels = torch.tensor(can_image_label, dtype=torch.float32)  #MSELoss -> torch.float32
    distribution_calculate(second_D, features, labels)
    test_p(second_D, features, labels, 0.5)
