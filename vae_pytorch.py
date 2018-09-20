'''
reference
https://arxiv.org/abs/1312.6114
https://github.com/wohlert/semi-supervised-pytorch
'''


import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
cuda = torch.cuda.is_available()
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    '''
    distribution q_phi (z|x) that infer p(z|x)
    '''
    def __init__(self,dims):
        super(Encoder,self).__init__()
        [x_dim, h_dim, z_dim] = dims # h_dim is list

        nodes = [x_dim, *h_dim]
        linear_layers = [ nn.Linear(nodes[idx-1],nodes[idx] ) for idx in range(1,len(nodes))] # linear_layers is list which is consist of nn.Linear()
        
        self.hidden = nn.ModuleList(linear_layers) # pytorch does not see nn.module's parameter in python list.

        self.mu = nn.Linear(h_dim[-1], z_dim)
        self.log_var = nn.Linear(h_dim[-1], z_dim)

    def forward(self,x): # nn architecture
        for layer in self.hidden:
            x = F.relu(layer(x))
        
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x)) # f(x) = log(1+expx)
        
        return mu , log_var
        

        
class Sample(nn.Module):
    def __init__(self):
        super(Sample,self).__init__()
        
    def reparametrize(self,mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad = False)
        
        if mu.is_cuda:
            epsilon = epsilon.cuda()
            
        # log_var*0.5 = log_std ' exp(log_std) = std
        # log_var 
        std = log_var.mul(0.5).exp_() 
        
        #z= mu + std*epsilon
        _z = mu.addcmul(std, epsilon)

        return _z              
        
    def forward(self,x):
        mu, log_var = x

        z = self.reparametrize(mu, log_var)

        return z


        
class Decoder(nn.Module):
    '''
    dist. p_theta (x|z)
    '''
    
    def __init__(self,dims):
        super(Decoder,self).__init__()
        [z_dim, h_dim, x_dim] = dims #h_dim is reversed 
        
        nodes = [z_dim, *h_dim]
        linear_layer = [ nn.Linear(nodes[idx-1],nodes[idx])  for idx in range(1, len(nodes))]
        self.hidden = nn.ModuleList(linear_layer)
        
        self.last_layer = nn.Linear(h_dim[-1], x_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        for layer in self.hidden:

            x = F.relu(layer(x))

        reconstruction = self.sigmoid(self.last_layer(x))
        return reconstruction
        

        
def log_gaussian(x, mu, log_var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and log_var evaluated at x.

    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    mu=torch.tensor(mu).type(torch.float32)
    log_var=torch.tensor(log_var).type(torch.float32)    
    
    log_pdf = - 0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu)**2 / (2 * torch.exp(log_var))

    return torch.sum(log_pdf, dim=-1)

    
class vae(nn.Module):
    
    def __init__(self,dims,p_param):
        super(vae,self).__init__()
        
        self.p_param = p_param
        [x_dim, h_dim, z_dim] = dims
        self.z_dim = z_dim

        self.encoder = Encoder([x_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim, list(reversed(h_dim)), x_dim])
        self.kl_divergence = 0

        for m in self.modules(): # Returns an iterator over all modules in the network.
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def _kld(self, z, q_param):

        (mu_hat, log_var_hat) = q_param
        qz = log_gaussian(z, mu_hat, log_var_hat)  
        
        (mu, log_var) = self.p_param
        pz = log_gaussian(z, mu, log_var)
     
        kl = qz - pz
        
        return kl
                    

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.

        :param x: input data
        :return: reconstructed input
        """
        z_mu, z_log_var = self.encoder(x)

        z = Sample()((z_mu, z_log_var))

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))
        x_mu = self.decoder(z)

        return x_mu

    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)
   
        
# use custom BCE to sum up with Regularization term 
def binary_cross_entropy(pred_y,y):
    return -torch.sum(y*torch.log(pred_y+1e-8)+ (1-y)*torch.log(1-pred_y + 1e-8), dim=-1)                                                                                                                                                                                                             
      

from torchvision import datasets, transforms

batch_size=64
test_batch_size=1000

### load data
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, 
                          transform=transforms.Compose([ 
                          transforms.ToTensor()
                          ])), batch_size=batch_size, shuffle=True, **kwargs) 

### 

(mu, log_var) = (0, 0)
p_param = (mu, log_var)
dims=([784, [256, 128], 32])

model = vae(dims,p_param)

optimizer = torch.optim.Adam(model.parameters())

for epoch in range(50):
    model.train()
    total_loss = 0
    for (batch_x, _) in train_loader:
        batch_x=batch_x.view(-1,28*28)

        if cuda: batch_x = batch_x.cuda(device=0)

        reconstruction = model(batch_x)
        
        likelihood = -binary_cross_entropy(reconstruction, batch_x)
        elbo = likelihood - model.kl_divergence
        L = -torch.mean(elbo)

        L.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += L.data[0]

    m = len(train_loader)

    if epoch % 1 == 0:
        print("Epoch: {} \t L: {}".format(epoch,total_loss/m))

model.eval()
x_mu = model.sample(Variable(torch.randn(16, 32)))

f, axarr = plt.subplots(1, 16, figsize=(18, 12))

samples = x_mu.data.view(-1, 28, 28).numpy()

for i, ax in enumerate(axarr.flat):
    ax.imshow(samples[i])
    ax.axis("off")        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        