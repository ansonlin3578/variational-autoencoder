import torch
import numpy as np
import cv2
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pdb
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models import AE
from models import VAE
## 11110COM 526000 Deep Learning HW2:Variational Autoencoder
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Hyperparameters
random_seed = 123
learning_rate = 0.005
num_epochs = 5
batch_size = 4

## Don't change the below two functions (compute_PSNR, compute_SSIM)!!
def compute_PSNR(img1, img2): ## 請輸入範圍在0~1的圖片!!!
    # Compute Peak Signal to Noise Ratio (PSNR) function
    # img1 and img2 > [0, 1] 
    
    img1 = torch.as_tensor(img1, dtype=torch.float32)# In tensor format!!
    img2 = torch.as_tensor(img2, dtype=torch.float32)
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1 / torch.sqrt(mse))

def compute_SSIM(img1, img2): ## 請輸入範圍在0~1的圖片!!!
    # Compute Structure Similarity (SSIM) function
    # img1 and img2 > [0, 1]
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

## Feel free to adjust the codes in the hw2_template.py !!!

## Read the data
def npy_loader(path):
    return torch.from_numpy(np.load(path))
data = npy_loader(r"C:\Users\User\Desktop\碩一上修課資料\深度學習_林嘉文\HW2\eye\data.npy")
label = npy_loader(r"C:\Users\User\Desktop\碩一上修課資料\深度學習_林嘉文\HW2\eye\label.npy")

data = data.permute(0,3,1,2) #convert (50,50,3) -> (3, 50, 50)
data_gpu, label_gpu = data.to(device), label.to(device)

dataset = TensorDataset(data_gpu , label_gpu)
loader = DataLoader(dataset, batch_size=4) 
# print(data[0])
# Checking the dataset
for images, labels in loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

## Your training process, loss function and save the torch model in (.pth) format.

class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :50, :50]

def plot_loss(losses , losses_avg):
    iter_len = len(losses)
    x_axis = np.array([i for i in range(iter_len)])
    plt.plot(x_axis , losses)
    plt.plot(x_axis , losses_avg)

    plt.xlabel("Iterations")
    pic_name = "VAE_Loss_low_code.png"
    plt.ylabel("Loss")

    plt.title("Loss per iteration")
    plt.legend(['losses', 'losses_avg'], loc="best")
    plt.savefig(pic_name)
    plt.close()

# #create a model from 'AE' autoencoder class
AE_model = AE()
AE_model.to(device)
AE_model.double()
AE_optimizer = optim.Adam(AE_model.parameters(), lr=learning_rate)
# mean-squared error loss
AE_criterion = nn.MSELoss()
########################################## AE Training ##################################################
print(AE_model)
print("AE_model start training!!")

AE_losses , AE_losses_avg = [], []
for epoch in range(num_epochs):
    loss = 0
    for idx , batches in enumerate(loader):
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        AE_optimizer.zero_grad()

        # compute reconstructions
        outputs = AE_model(batches[0])
    
        # compute training reconstruction loss
        train_loss = AE_criterion(outputs, batches[0])

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        AE_optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
        AE_losses.append(train_loss.item())
        AE_losses_avg.append(sum(AE_losses)/len(AE_losses))

    # compute the epoch training loss
    loss = loss / len(loader)

    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, loss))

# # #create a model from "VAE"autoencoder class
VAE_model = VAE().to(device)
VAE_model.double()
VAE_optimizer = optim.Adam(VAE_model.parameters(), lr=learning_rate)
VAE_criterion = nn.MSELoss()
#/////////////////////////////////////////// VAE Training //////////////////////////////////////////
print(VAE_model)
print("VAE_model start training!!")
VAE_losses , VAE_losses_avg = [], []
for epoch in range(num_epochs):
    loss = 0
    for idx , batches in enumerate(loader):
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        VAE_optimizer.zero_grad()

        # compute reconstructions
        outputs = VAE_model(batches[0])
    
        # compute training reconstruction loss
        train_loss = VAE_criterion(outputs, batches[0])

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        VAE_optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
        VAE_losses.append(train_loss.item())
        VAE_losses_avg.append(sum(VAE_losses)/len(VAE_losses))

    # compute the epoch training loss
    loss = loss / len(loader)

    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, loss))

def tensor_to_np(tensor):
    temp = tensor.permute(0, 2, 3, 1)
    arr = temp.cpu().detach().numpy()
    arr_per_batch = []
    for i in range(batch_size):
        arr_per_batch.append(arr[i])
    return arr_per_batch

def choose_img(arr_per_batch, choose_idx: int):
    return arr_per_batch[choose_idx]

def save_img(choose_reco, choose_label, start_batch_idx , img_select , choose_idx):
    for i in range(start_batch_idx, start_batch_idx + 5):
        arr_per_batch = tensor_to_np(choose_reco[i])
        plt.subplot(1,5,i - start_batch_idx+ 1)
        img = choose_img(arr_per_batch, img_select-1)
        img = img[:,:,::-1]
        plt.imshow((img * 255).astype(np.uint8))
    plt.title("VAE_{}_idx_reconstructed".format(choose_idx), loc="center")
    plt.xlabel("VAE_{}".format(choose_label[0][img_select-1]) , loc="center")
    plt.savefig("VAE_{}_idx_reconstructed".format(choose_idx))
    plt.close()

comparison_sets = []
for img_batch, label in loader:
    same_gaussion = torch.randn(4, 256, device=device, dtype=torch.double)
    AE_reco_batch = AE_model.comparison(img_batch, same_gaussion)
    VAE_reco_batch = VAE_model.comparison(img_batch, same_gaussion)
    AE_reco_arr = tensor_to_np(AE_reco_batch)
    VAE_reco_arr = tensor_to_np(VAE_reco_batch)
    comparison_sets.append((AE_reco_arr, VAE_reco_arr))
    break
def compare_img(AE_gen, VAE_gen):
    for i in range(batch_size):
        AE_img, VAE_img = AE_gen[i], VAE_gen[i]
        data_tensor = data[i].permute(1, 2, 0)
        ori_img = data_tensor.numpy()
        AE_img = AE_img[:,:,::-1]
        VAE_img = VAE_img[:,:,::-1]
        ori_img = ori_img[:,:,::-1]
        plt.subplot(1,3,1)
        plt.title("original")
        plt.imshow((ori_img * 255).astype(np.uint8))
        plt.subplot(1,3,2)
        plt.title("AE")
        plt.imshow((AE_img * 255).astype(np.uint8))
        plt.subplot(1,3,3)
        plt.title("VAE")
        plt.imshow((VAE_img * 255).astype(np.uint8))
        plt.xlabel("Img ID is {}".format(i) , loc="center")
        plt.savefig("VAE_AE_comparison_ID_{}".format(i))
        plt.close()
for idx in range(len(comparison_sets)):
    compare_img(comparison_sets[idx][0] , comparison_sets[idx][1])

## Your average PSNR, average SSIM on 1476 images and visualization results.
# psnr_total = []
# for i in range(len(reconstructed_imgs)):
#     psnr = compute_PSNR(data[i], reconstructed_imgs[i])
#     psnr_float = psnr.numpy()
#     psnr_total.append(float(psnr_float))
# print("PSNR Mean : {:.6f}".format(sum(psnr_total)/len(psnr_total)))

# ssim_total = []
# for i in range(len(reconstructed_imgs)):
#     img_ori , img_reco = data[i].numpy() , reconstructed_imgs[i].numpy()
#     ssim = compute_SSIM(img_ori, img_reco)
#     ssim_total.append(float(ssim))
# print("SSIM Mean : {:.6f}".format(sum(ssim_total)/len(ssim_total)))

