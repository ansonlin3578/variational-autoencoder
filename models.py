import torch
from torch import nn
import torch.optim as optim
import pdb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, dilation=1)  
        # conv layer (depth from 32 --> 64), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)
        #Hout​ =(Hin​ −1)×stride[0]−2×padding[0]+kernel_size[0]
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.encfc = nn.Linear(in_features=64*12*12, out_features=256)
        self.relu = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        #H out =(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        self.t_conv1 = nn.ConvTranspose2d(64, 32, 2, stride=2,  dilation=1, output_padding=1)
        self.t_conv2 = nn.ConvTranspose2d(32, 3, 2, stride=2, dilation=1, output_padding=0)
        self.t_conv3 = nn.ConvTranspose2d(64, 32, 2, stride=1, dilation=1, output_padding=0, padding=0)
        self.t_conv4 = nn.ConvTranspose2d(32, 3, 3, stride=1, dilation=1, output_padding=0, padding=1)
        self.decfc = nn.Linear(in_features=256, out_features=64*12*12)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = self.conv1(x)   #in 50 out 50
        x = self.relu(x)
        x , indices_1= self.pool(x) #in 50 out 25
        # add second hidden layer
        x = self.conv2(x)   #in 25 out 25
        x = self.relu(x)
        x , indices_2= self.pool(x)  #in 25 out 12
        x = x.view(-1, 64*12*12)
        x = self.encfc(x)

        ## decode ##
        x = self.decfc(x)
        x = x.view(-1, 64, 12, 12)
        x = self.unpool(x , indices_2)  
        x = self.relu(x)
        x = self.t_conv3(x)
        x = self.unpool(x , indices_1) 
        x = self.t_conv4(x)
        x = self.sigmoid(x)
        return x
    def add_gaussion(self, x):
        x = self.conv1(x)   #in 50 out 50
        x = self.relu(x)
        x , indices_1= self.pool(x) #in 50 out 25
        # add second hidden layer
        x = self.conv2(x)   #in 25 out 25
        x = self.relu(x)
        x , indices_2= self.pool(x)  #in 25 out 12
        x = x.view(-1, 64*12*12)
        x = self.encfc(x)
        noise = torch.randn(4, 256, device=device, dtype=torch.double)
        x = x + noise
        ## decode ##
        x = self.decfc(x)
        x = x.view(-1, 64, 12, 12)
        x = self.unpool(x , indices_2)  
        x = self.relu(x)
        x = self.t_conv3(x)
        x = self.unpool(x , indices_1) 
        x = self.t_conv4(x)
        x = self.sigmoid(x)
        return x
    def comparison(self, x, same_gaussion):
        x = self.conv1(x)   #in 50 out 50
        x = self.relu(x)
        x , indices_1= self.pool(x) #in 50 out 25
        # add second hidden layer
        x = self.conv2(x)   #in 25 out 25
        x = self.relu(x)
        x , indices_2= self.pool(x)  #in 25 out 12
        x = x.view(-1, 64*12*12)
        x = self.encfc(x)
        x = x + same_gaussion
        ## decode ##
        x = self.decfc(x)
        x = x.view(-1, 64, 12, 12)
        x = self.unpool(x , indices_2)  
        x = self.relu(x)
        x = self.t_conv3(x)
        x = self.unpool(x , indices_1) 
        x = self.t_conv4(x)
        x = self.sigmoid(x)
        return x
##############################################################################################################

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, dilation=1)  
        # conv layer (depth from 32 --> 64), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)
        #Hout​ =(Hin​ −1)×stride[0]−2×padding[0]+kernel_size[0]
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.encFC1 = nn.Linear(in_features=64*12*12, out_features=256)
        self.encFC2 = nn.Linear(in_features=64*12*12, out_features=256)
        self.relu = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        #H out =(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        self.t_conv1 = nn.ConvTranspose2d(64, 32, 2, stride=2,  dilation=1, output_padding=1)
        self.t_conv2 = nn.ConvTranspose2d(32, 3, 2, stride=2, dilation=1, output_padding=0)
        self.t_conv3 = nn.ConvTranspose2d(64, 32, 2, stride=1, dilation=1, output_padding=0, padding=0)
        self.t_conv4 = nn.ConvTranspose2d(32, 3, 3, stride=1, dilation=1, output_padding=0, padding=1)
        self.decFC = nn.Linear(in_features=256, out_features=64*12*12)

    def reparameterize(self, mu, logVar):
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        pdb.set_trace()
        x = self.conv1(x)   #in 50 out 50
        x = self.relu(x)
        x , indices_1= self.pool(x) #in 50 out 25
        # add second hidden layer
        x = self.conv2(x)   #in 25 out 25
        x = self.relu(x)
        x , indices_2= self.pool(x) #in 25 out 12
        x = x.view(-1, 64*12*12)
        mu  = self.encFC1(x)  
        logvar = self.encFC2(x)
        z = self.reparameterize(mu, logvar)

        ## decode ##
        z = self.decFC(z)
        z = z.view(-1, 64, 12, 12)
        z = self.unpool(z , indices_2) 
        z = self.relu(z)
        z = self.t_conv3(z)
        z = self.unpool(z , indices_1) 
        z = self.t_conv4(z)
        z = self.sigmoid(z)
        return z
    def add_gaussion(self, x):
        x = self.conv1(x)   #in 50 out 50
        x = self.relu(x)
        x , indices_1= self.pool(x) #in 50 out 25
        # add second hidden layer
        x = self.conv2(x)   #in 25 out 25
        x = self.relu(x)
        x , indices_2= self.pool(x)  #in 25 out 12
        x = x.view(-1, 64*12*12)
        mu  = self.encFC1(x)  
        logvar = self.encFC2(x)
        z = self.reparameterize(mu, logvar)
        noise = torch.randn(4, 256, device=device, dtype=torch.double)
        z = z + noise
        ## decode ##
        z = self.decFC(z)
        z = z.view(-1, 64, 12, 12)
        z = self.unpool(z , indices_2) 
        z = self.relu(z)
        z = self.t_conv3(z)
        z = self.unpool(z , indices_1) 
        z = self.t_conv4(z)
        z = self.sigmoid(z)
        return z
    def comparison(self, x, same_gaussion):
        x = self.conv1(x)   #in 50 out 50
        x = self.relu(x)
        x , indices_1= self.pool(x) #in 50 out 25
        # add second hidden layer
        x = self.conv2(x)   #in 25 out 25
        x = self.relu(x)
        x , indices_2= self.pool(x)  #in 25 out 12
        x = x.view(-1, 64*12*12)
        mu  = self.encFC1(x)  
        logvar = self.encFC2(x)
        z = self.reparameterize(mu, logvar)
        z = z + same_gaussion
        ## decode ##
        z = self.decFC(z)
        z = z.view(-1, 64, 12, 12)
        z = self.unpool(z , indices_2) 
        z = self.relu(z)
        z = self.t_conv3(z)
        z = self.unpool(z , indices_1) 
        z = self.t_conv4(z)
        z = self.sigmoid(z)
        return z
class high_latent(nn.Module):
    def __init__(self):
        super(high_latent, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, dilation=1)  
        # conv layer (depth from 32 --> 64), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, return_indices=True)
        #Hout​ =(Hin​ −1)×stride[0]−2×padding[0]+kernel_size[0]
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=1, padding=0)
        self.encfc = nn.Linear(in_features=64*12*12, out_features=256)
        self.relu = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        #H out =(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        self.t_conv1 = nn.ConvTranspose2d(64, 32, 2, stride=2,  dilation=1, output_padding=1)
        self.t_conv2 = nn.ConvTranspose2d(32, 3, 2, stride=2, dilation=1, output_padding=0)
        self.t_conv3 = nn.ConvTranspose2d(64, 32, 3, stride=1, dilation=1, output_padding=0, padding=1)
        self.t_conv4 = nn.ConvTranspose2d(32, 3, 3, stride=1, dilation=1, output_padding=0, padding=1)
        self.decfc = nn.Linear(in_features=256, out_features=64*12*12)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = self.conv1(x)   #in 50 out 50
        x = self.relu(x)
        x , indices_1= self.pool(x) #in 50 out 49
        # add second hidden layer
        x = self.conv2(x)   #in 49 out 49
        x = self.relu(x)
        x , indices_2= self.pool(x)  #in 49 out 48

        ## decode ##
        x = self.unpool(x , indices_2)  #in 48 out 49
        x = self.relu(x)
        x = self.t_conv3(x)          #in 49 out 50
        x = self.unpool(x , indices_1) 
        x = self.t_conv4(x)
        x = self.sigmoid(x)
        return x