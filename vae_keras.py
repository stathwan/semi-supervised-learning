"""
VAE(Variational Autoencoder)입니다. 
주석을 한국어로 작성하여 한국어 사용자들이 이해하기 쉽게 하였습니다.
궁금한점이나/수정이 필요한 부분은 알려주세요
이메일: stat_hwan@naver.com
"""

# import module, 모듈 불러오기 
import numpy as np
import math
import matplotlib.pyplot as plt

from keras.layers import Dense, Input, Lambda
from keras.models import Model 
from keras import backend as K 

from keras import metrics
from keras.datasets import mnist
from keras.optimizers import adam


# hyperparameter, 초모수설정
p_parm=(0.,0.) #mu, log_var
dims=[784, [256, 128], 2]
batch_size=128
epochs=10


#load data, 데이터 불러오기
(x_train, _ ), (x_test, y_test ) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(-1,784)
x_test  = x_test.reshape(-1,784)



def Encoder(input_dim):
    '''
    distribution q_phi (z|x) that infer p(z|x)
    '''
    z_dims = dims[-1]
    input_shape = dims[0]
    hidden_nodes = dims[1] # list 
    
    x= Input((input_shape,))
    encoder_h= Dense(hidden_nodes[0] , activation='relu')(x)
    encoder_h= Dense(hidden_nodes[1], activation='relu')(encoder_h)

    mu_hat= Dense(z_dims,activation='linear')(encoder_h) # default latent_dim=2 
    log_variance_hat= Dense(z_dims, activation='linear')(encoder_h) 
    
    return Model(x, (mu_hat,log_variance_hat))



# Sampling layer by Lambda 
def Sample(q_parm):
    mu_hat,log_var_hat = q_parm
    z_dim=int(mu_hat.shape[1])
    _epsilon= K.random_normal(shape=(z_dim,), mean=0., stddev=1.)
    z=mu_hat+K.exp(log_var_hat / 2)*_epsilon
    return z 


def Decoder(input_dim):
    '''
    dist. p_theta (x|z)
    '''
    z_dims = dims[-1]
    input_shape = dims[0]
    hidden_nodes = list(reversed(dims[1])) # list 
    z=Input((z_dims,))
    
    decoder_h= Dense(hidden_nodes[0], activation='relu', name='decoder_h1')(z) # to make generator
    decoder_h= Dense(hidden_nodes[1] ,activation='relu',  name='decoder_h2')(decoder_h)
    x_hat    = Dense(input_shape, activation='sigmoid', name='decoder_h3')(decoder_h)
    
    return Model(z, x_hat)


encoder=Encoder(dims)
decoder=Decoder(dims)

# latent z, 잠재변수 z  
batch_x=Input((dims[0],))
mu_hat, log_var_hat = encoder(batch_x)
z     = Lambda(Sample, output_shape=(dims[-1],))([mu_hat,log_var_hat]) # q_parm = mu_hat,log_var_hat ;; latent dim = dims[-1]
x_hat = decoder(z)
vae=Model(batch_x, x_hat)


class VAEloss():
    def __init__(self, p_parm ,z, mu_hat, log_var_hat):
        self.p_parm = p_parm
        self.z = z
        self.mu_hat = mu_hat
        self.log_var_hat = log_var_hat
        
    def log_gaussian(self, x, mu, log_var):
        """
        :return: log N(x|µ,σ) for loss
        """
        log_pdf = - 0.5 * math.log(2 * math.pi) - log_var / 2 - (x-mu)**2 / (2 * K.exp(log_var))
        return K.sum(log_pdf,axis=-1)
    
    def kl_divergence(self):
        """
        :return: kldivergence_loss for q_param p_param
        """
        qz = self.log_gaussian(self.z, self.mu_hat, self.log_var_hat)
        mu, log_var = self.p_parm
        pz = self.log_gaussian(self.z, mu, log_var)
        kl = qz - pz
        
        return kl
    
    def likelihood(y, y_pred):
        return -metrics.binary_crossentropy(y, y_pred)*786

    def vaeloss(self,x, x_hat):
        _x     = K.flatten(x)
        _x_hat = K.flatten(x_hat)

        likelihood = self.likelihood(_x, _x_hat)
        KLDivergence = self.kl_divergence()
        elbo = likelihood - KLDivergence
        loss = -K.mean(elbo)
        return loss
    

loss=VAEloss(p_parm ,z, mu_hat, log_var_hat).vaeloss

vae.compile(loss = loss, optimizer=adam(0.0002,0.5))
    
#fit & save model
vae.fit(x_train,x_train,shuffle=True, epochs=epochs, batch_size=batch_size)

#vae.save_weights('c:/data/vae/vae_model.hdf5')
#vae.load_weights('c:/data/vae/vae_model_latent1.hdf5')


#visuallize latent z
import matplotlib.colors as colors
import matplotlib.cm as cmx

mu_pred , _ = encoder.predict(x_test)
mu_pred= np.array(mu_pred)

cNorm  = colors.Normalize(vmin=0, vmax=len(np.unique(y_test)))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('hot') )
plt.scatter(mu_pred[:,0],mu_pred[:,1],c=scalarMap.to_rgba(y_test),cmap='a')

#predict
#a=decoder.predict(np.array([[0,0]]))
#aa=a.reshape(dims[0],dims[0]) # dims[0] = 0 
#plt.imshow(aa)

im_size=28
size=20
axis_x=np.linspace(-2,2,size)
axis_y=np.linspace(-2,2,size)

img=np.empty([size*im_size,size*im_size] )

for i,z_1, in enumerate(axis_x):
    for j,z_2, in enumerate(axis_y):
        x_pred=decoder.predict(np.array([[z_1,z_2]]))
        x_pred=x_pred.reshape(im_size,im_size)
        img[im_size*i :im_size*(i+1), im_size*j :im_size*(j+1)] = x_pred

plt.imshow(img)
