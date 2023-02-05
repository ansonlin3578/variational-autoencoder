import torch
import numpy as np
import cv2
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pdb
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models import high_latent
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

#///////////////////////////////////////Variational_Auto_encoder///////////////////////////////////////////////
## Your training process, loss function and save the torch model in (.pth) format.
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :50, :50]
#///////////////////////////////////////Auto_encoder///////////////////////////////////////////////

def plot_loss(losses , losses_avg):
    iter_len = len(losses)
    x_axis = np.array([i for i in range(iter_len)])
    plt.plot(x_axis , losses)
    plt.plot(x_axis , losses_avg)

    plt.xlabel("Iterations")
    pic_name = "Loss_low_code_epoch{}.png".format(num_epochs)
    plt.ylabel("Loss")

    plt.title("Loss per iteration")
    plt.legend(['losses', 'losses_avg'], loc="best")
    plt.savefig(pic_name)
    plt.close()

# #create a model from `AE` autoencoder class
model = high_latent().to(device)
model.double()
# #create a model from `VAE` autoencoder class
# VAE_model = VAE().to(device)
# model.double()

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# mean-squared error loss
criterion = nn.MSELoss()

##########################################Training##################################################
print(model)

losses , losses_avg = [], []
for epoch in range(num_epochs):
    loss = 0
    for idx , batches in enumerate(loader):
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        outputs = model(batches[0])
    
        # compute training reconstruction loss
        train_loss = criterion(outputs, batches[0])

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
        losses.append(train_loss.item())
        losses_avg.append(sum(losses)/len(losses))

    # compute the epoch training loss
    loss = loss / len(loader)

    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, loss))
# plot_loss(losses, losses_avg)
print("final loss average = {:.4f}".format(losses_avg[-1]))

def tensor_to_np(tensor):
    temp = tensor.permute(0, 2, 3, 1)
    arr = temp.cpu().detach().numpy()
    arr_per_batch = []
    for i in range(batch_size):
        arr_per_batch.append(arr[i])
    return arr_per_batch

def visualize(img_batch, model):
    """Draws original, encoded and decoded images"""
    # img[None] will have shape of (1, 32, 32, 3) which is the same as the model input
    reco_batch = model.forward(img_batch)

    plt.subplot(1,2,1)
    plt.title("Original")
    arr_per_batch = tensor_to_np(img_batch)
    for i in range(batch_size):
        img = (arr_per_batch[i] * 255).astype(np.uint8)
        img = img[:,:,::-1]
        plt.imshow(img)
        break

    plt.subplot(1,2,2)
    plt.title("Reconstructed")
    arr_per_batch = tensor_to_np(reco_batch)
    for i in range(batch_size):
        img = (arr_per_batch[i] * 255).astype(np.uint8)
        img = img[:,:,::-1]
        plt.imshow(img)
        break
    plt.savefig("test_high_code.png")
    plt.show()
for img_batch, lable in loader:
    visualize(img_batch, model)
    break
#/////////////////////////////////////////choose img to reconstruct///////////////////////////////////////////
# choose_id = [1,2,3,4,5,226,227,228,229,230,841,842,843,844,845,1471,1472,1473,1474,1475]
# indices = torch.tensor([i-1 for i in choose_id])
# choose_tensor = torch.index_select(data, 0, indices)
# choose_label = torch.index_select(label, 0, indices)
# choose_tensor_gpu , choose_label_gpu = choose_tensor.to(device) , choose_label.to(device)
# choose_dataset = TensorDataset(choose_tensor_gpu, choose_label_gpu)
# choose_loader = DataLoader(choose_dataset, batch_size=4)

# choose_reco = []
# choose_label = []
# def gnerate_with_noise(img_batch, model, label):
#     reco_result = []
#     gen_label_result = []
#     for _ in range(5):
#         each_result = model.add_gaussion(img_batch)
#         reco_result.append(each_result)
#         gen_label_result.append(label)
#     return reco_result, gen_label_result

# for img_batch, label in choose_loader:
#     reco_result, gen_label_result= gnerate_with_noise(img_batch, model, label)
#     for i in range(len(reco_result)):
#         choose_reco.append(reco_result[i])
#         choose_label.append(gen_label_result[i])

# gen_data = choose_reco[0]
# gen_label = choose_label[0]
# for i in range(1, len(choose_reco)):
#     merge_data = choose_reco[i]
#     merge_label = choose_label[i]
#     gen_data = torch.cat((gen_data, merge_data), 0)
#     gen_label = torch.cat((gen_label, merge_label), 0)
# gen_data = gen_data.cpu().detach().numpy()
# gen_label = gen_label.cpu().detach().numpy()
# # np.save(r"C:\Users\User\Desktop\碩一上修課資料\深度學習_林嘉文\HW2\gen_data.npy", gen_data)
# # np.save(r"C:\Users\User\Desktop\碩一上修課資料\深度學習_林嘉文\HW2\gen_label.npy", gen_label)

# def save_img(choose_reco, choose_label, start_batch_idx , img_select , choose_idx):
#     for i in range(start_batch_idx, start_batch_idx + 5):
#         arr_per_batch = tensor_to_np(choose_reco[i])
#         plt.subplot(1,5,i - start_batch_idx+ 1)
#         img = arr_per_batch[img_select-1]
#         img = img[:,:,::-1]
#         plt.imshow((img * 255).astype(np.uint8))
#     plt.title("{}_idx_reconstructed".format(choose_idx))
#     plt.xlabel("{}".format(choose_label[i][img_select-1]) , loc="center")
#     plt.savefig("{}_idx_reconstructed".format(choose_idx))
#     plt.close()
# save_img(choose_reco, choose_label, 0, 3, 3)
# save_img(choose_reco, choose_label, 5, 3, 227)
# save_img(choose_reco, choose_label, 10, 3, 841)
# save_img(choose_reco, choose_label, 20, 4, 1475)


# ## Your average PSNR, average SSIM on 1476 images and visualization results.
# reconstructed_imgs = []
# for img_batch, label in loader:
#     reco_batch = model.forward(img_batch)
#     reco_cpu = reco_batch.cpu().detach()
#     for img in reco_cpu:
#         reconstructed_imgs.append(img)

# psnr_total = []
# for i in range(len(reconstructed_imgs)):
#     psnr = compute_PSNR(data[i], reconstructed_imgs[i])
#     psnr_float = psnr.numpy()
#     psnr_total.append(float(psnr_float))
# print("PSNR Mean : {:.6f}".format(sum(psnr_total)/len(psnr_total)))

# ssim_total = []
# for i in range(len(reconstructed_imgs)):
#     img_ori , img_reco = data[i].permute(1,2,0).numpy() , reconstructed_imgs[i].permute(1,2,0).numpy()
#     ssim = compute_SSIM(img_ori, img_reco)
#     ssim_total.append(float(ssim))
# print("SSIM Mean : {:.6f}".format(sum(ssim_total)/len(ssim_total)))

