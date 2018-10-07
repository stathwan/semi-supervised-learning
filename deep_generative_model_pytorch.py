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
from itertools import cycle


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

class Classifier(nn.Module):
    def __init__(self, dims):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        self.dense = nn.Linear(x_dim, h_dim)
        self.logits = nn.Linear(h_dim, y_dim)

    def forward(self, x):
        x = F.relu(self.dense(x))
        x = F.softmax(self.logits(x), dim=-1)
        return x

        
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

    
class DeepGenerativeModel(nn.Module):
    
    def __init__(self,dims,p_param):
        super(DeepGenerativeModel,self).__init__()
        
        self.p_param = p_param
        [x_dim, self.y_dim, h_dim, self.z_dim] = dims

        self.encoder = Encoder([x_dim+ self.y_dim, h_dim, self.z_dim])
        self.decoder = Decoder([self.z_dim+ self.y_dim, list(reversed(h_dim)), x_dim])
        self.classifier = Classifier([x_dim, h_dim[0], self.y_dim])
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
                    

    def forward(self, x, y):
        # Add label and data and generate latent variable
        z_mu, z_log_var = self.encoder(torch.cat([x, y], dim=1))
        z = Sample()((z_mu, z_log_var))
        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        # Reconstruct data point from latent data and label
        x_mu = self.decoder(torch.cat([z, y], dim=1))

        return x_mu

    def classify(self, x):
        logits = self.classifier(x)
        return logits

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an x.
        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        y = y.float()
        x = self.decoder(torch.cat([z, y], dim=1))
        return x
   
              

class Stochastic_variational_inference(nn.Module):
    """
    Stochastic variational inference (SVI).
    """

    
    def __init__(self, model, likelihood=F.binary_cross_entropy):
        """
        Initialises a new SVI optimizer for semi-
        supervised learning.
        :param model: semi-supervised model to evaluate
        :param likelihood: p(x|y,z) for example BCE or MSE
        :param sampler: sampler for x and y, e.g. for Monte Carlo
        :param beta: warm-up/scaling of KL-term
        """
        super(Stochastic_variational_inference, self).__init__()
        self.model = model
        self.likelihood = likelihood

    def log_sum_exp(self, tensor, dim=-1, sum_op=torch.sum):
        """
        Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
        :param tensor: Tensor to compute LSE over
        :param dim: dimension to perform operation over
        :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
        :return: LSE
        """
        max, _ = torch.max(tensor, dim=dim, keepdim=True)
        return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max

    def sampler(self,elbo):
        elbo = elbo.view(1, 1, -1)
        elbo = torch.mean(self.log_sum_exp(elbo, dim=1, sum_op=torch.mean), dim=0)
        return elbo.view(-1)

    def log_standard_categorical(self,p):
        """
        Calculates the cross entropy between a (one-hot) categorical vector
        and a standard (uniform) categorical distribution.
    
        :param p: one-hot categorical distribution
        :return: H(p, u)
        """
        # Uniform prior over y
        prior = F.softmax(torch.ones_like(p), dim=1)
        prior.requires_grad = False
    
        cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)
    
        return cross_entropy
    
    def enumerate_discrete(self, x, y_dim):
        """
        Generates a `torch.Tensor` of size batch_size x n_labels of
        the given label.
    
        Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                           [0, 1, 0]])
        :param x: tensor with batch size to mimic
        :param y_dim: number of total labels
        :return variable
        """
        def batch(batch_size, label):
            labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
            y = torch.zeros((batch_size, y_dim))
            y.scatter_(1, labels, 1)
            return y.type(torch.LongTensor)
    
        batch_size = x.size(0)
        generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])
    
        if x.is_cuda:
            generated = generated.cuda()
    
        return Variable(generated.float())
    
    def forward(self, x, y=None):
        is_labelled = False if y is None else True

        # Prepare for sampling
        xs, ys = (x, y)

        # Enumerate choices of label
        if not is_labelled:
            ys = self.enumerate_discrete(xs, self.model.y_dim)
            xs = xs.repeat(self.model.y_dim, 1)
            
        reconstruction = self.model(xs, ys)

        # p(x|y,z)

        likelihood = -self.likelihood(reconstruction, xs)

        # p(y)
        prior = -self.log_standard_categorical(ys)
         
        # Equivalent to -L(x, y)
        elbo = likelihood + prior - self.model.kl_divergence        
            
        L = self.sampler(elbo)

        if is_labelled:
            return torch.mean(L)
       
        logits = self.model.classify(x)

        L = L.view_as(logits.t()).t()

        # Calculate entropy H(q(y|x)) and sum over all labels
        H = -torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)
        L = torch.sum(torch.mul(logits, L), dim=-1)

        # Equivalent to -U(x)
        U = L + H
        
        return torch.mean(U)
        
        
### load data

# Only use 10 labelled examples per class
# The rest of the data is unlabelled.
#from datautils import get_mnist
import numpy as np
n_labels=10

def get_mnist(location="./", batch_size=64, labels_per_class=100):
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms

    def onehot(k):
        """
        Converts a number to its one-hot or 1-of-k representation
        vector.
        :param k: (int) length of vector
        :return: onehot function
        """
        def encode(label):
            y = torch.zeros(k)
            if label < k:
                y[label] = 1
            return y
            
        return encode
        
    flatten_bernoulli = lambda x: transforms.ToTensor()(x).view(-1).bernoulli()

    mnist_train = MNIST(location, train=True, download=True,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels))
    mnist_valid = MNIST(location, train=False, download=True,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels))

    
    def get_sampler(labels, n=None):
        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler

    # Dataloaders for MNIST
    labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=0, pin_memory=cuda,
                                           sampler=get_sampler(mnist_train.train_labels.numpy(), labels_per_class))
    unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=0, pin_memory=cuda,
                                             sampler=get_sampler(mnist_train.train_labels.numpy()))
    validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, num_workers=0, pin_memory=cuda,
                                             sampler=get_sampler(mnist_valid.test_labels.numpy()))

    return labelled, unlabelled, validation


labelled, unlabelled, validation = get_mnist(location="./", batch_size=64, labels_per_class=10)
alpha = 0.1 * len(unlabelled) / len(labelled)

# use custom BCE to sum up with Regularization term 
def binary_cross_entropy(pred_y,y):
    return -torch.sum(y * torch.log(pred_y + 1e-8) + (1 - y) * torch.log(1 - pred_y + 1e-8), dim=1)

### build model
p_param = (mu, log_var) = (0, 0)
dims=([784, 10, [256, 128], 32])

model = DeepGenerativeModel(dims,p_param)
optimizer = torch.optim.Adam(model.parameters())


if cuda: model = model.cuda()

elbo = Stochastic_variational_inference(model, likelihood=binary_cross_entropy)


for epoch in range(10):
    model.train()
    total_loss, accuracy = (0, 0)

    for (x, y), (u, _) in zip(cycle(labelled), unlabelled):
        # Wrap in variables
        x, y, u = Variable(x), Variable(y), Variable(u)

        if cuda:
            # They need to be on the same device and be synchronized.
            x, y = x.cuda(device=0), y.cuda(device=0)
            u = u.cuda(device=0)
            
        L = -elbo(x, y)
        U = -elbo(u)

   
            
        # Add auxiliary classification loss q(y|x)
        logits = model.classify(x)
        
        # Regular cross entropy
        classication_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

        J_alpha = L - alpha * classication_loss + U

        
        J_alpha.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += J_alpha.item()
        accuracy += torch.mean((torch.max(logits, 1)[1].data == torch.max(y, 1)[1].data).float())             
    
    if epoch % 1 == 0:
        model.eval()
        m = len(unlabelled)
        print("Epoch: {}".format(epoch))
        print("[Train]\t\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))

        total_loss, accuracy = (0, 0)
        for x, y in validation:
            x, y = Variable(x), Variable(y)

            if cuda:
                x, y = x.cuda(device=0), y.cuda(device=0)

            L = -elbo(x, y)
            U = -elbo(x)

            logits = model.classify(x)
            classication_loss = -torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

            J_alpha = L + alpha * classication_loss + U

            total_loss += J_alpha.item()

            _, pred_idx = torch.max(logits, 1)
            _, lab_idx = torch.max(y, 1)
            accuracy += torch.mean((torch.max(logits, 1)[1].data == torch.max(y, 1)[1].data).float())

        m = len(validation)
        print("[Validation]\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))
        
        


model.eval()
x_mu = model.sample(Variable(torch.randn(16, 32)))

f, axarr = plt.subplots(1, 16, figsize=(18, 12))

samples = x_mu.data.view(-1, 28, 28).numpy()

for i, ax in enumerate(axarr.flat):
    ax.imshow(samples[i])
    ax.axis("off")        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        