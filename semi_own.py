


#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        self.h1 = nn.Linear(784,256)
        self.h2 = nn.Linear(256,128)
        self.h3 = nn.Linear(128,10)        
        
    def forward(self,x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.softmax(self.h3(x), dim=1)        
        return x
        
class nn_loss(nn.Module):
    def __init__(self):
        super(nn_loss, self).__init__()
        print('cal loss')
        self.null = torch.Tensor([1,0,0,0,0,0,0,0,0,0])
        
    def cross_entropy(self,y,y_pred):
        return -torch.sum(y * torch.log(y_pred + 1e-8), dim=1)    

    def discrete_kld(self,y_pred):
        # D_kl = sumation p(x)log(p(x)/q(x))
        # optimal y dist. [1,0,0,0,0,0,0,0,0,0]
        y_pred = torch.sort(y_pred, descending=True, dim=1)[0]
        y =  self.null 
        return torch.sum(y_pred*torch.log((y_pred+1e-8)/(y+1e-8)), dim=1)
        
model = SCNN()
loss_f = nn_loss()
optimizer = torch.optim.Adam(model.parameters())

        
from datautils import get_mnist 
labelled, unlabelled, validation = get_mnist(location="./", batch_size=64, labels_per_class=10)

from itertools import cycle
from torch.autograd import Variable


for epoch in range(10):
    model.train()
    total_loss, accuracy = (0,0)
    
    for (x, y), (u, _) in zip(cycle(labelled), unlabelled):
        x, y, u = Variable(x), Variable(y), Variable(u)
        optimizer.zero_grad()
        y_pred_label = model(x)
        label_loss = loss_f.cross_entropy(y,y_pred_label)

        y_pred_unlabel = model(u)
        unlabel_loss = loss_f.discrete_kld(y_pred_unlabel)
    
        loss = torch.mean(label_loss) + torch.mean(unlabel_loss)*0.1
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        accuracy += torch.mean((torch.max(y_pred_label, 1)[1].data == torch.max(y, 1)[1].data).float())    

    if epoch % 1 == 0:
        model.eval()
        m = len(unlabelled)
        
        print("Epoch: {}".format(epoch))
        print("[Train]\t\t total_loss: {:.5f}, accuracy: {:.5f}".format(total_loss / m, accuracy / m))        

        total_loss, accuracy = (0,0)
        for x, y in validation:
            x, y = Variable(x), Variable(y)
            
            y_pred = model(x)
            label_loss = loss_f.cross_entropy(y,y_pred)
            unlabel_loss = loss_f.discrete_kld(y_pred)

            loss = torch.mean(label_loss) + torch.mean(unlabel_loss)*0.1    

            total_loss += loss.item()    
            
#            pred_idx = torch.max(y_pred, 1)[1]
#            label_idx = torch.max(y, 1)[1]
            
            accuracy += torch.mean((torch.max(y_pred,1)[1].data == torch.max(y,1)[1].data).float())
            
        m = len(validation)
        print("[Validation]\t total_loss: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))
            