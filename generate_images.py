from keras.models import load_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

plt.style.use(['dark_background'])


EPOCHS = 40

(_,_),(test,_) = mnist.load_data(path='data')
test = test.reshape((len(test), np.prod(test.shape[1:])))

till = 100


for epoch in tqdm(range(1,EPOCHS+1)):
    #os.system('cls')
    # Loading saved model
    saved_model = load_model(os.path.join('saved_models','autoencoder-epoch_{}.h5'.format(epoch)))

    # Getting Predictions for test data
    pred = saved_model.predict(test)
    pred = pred[0:till]

    # Memory Release
    del saved_model

    # Saving Images according to folder
    for i in range(len(pred)):
        if os.path.isdir(f'images/{i}') == False:
            os.makedirs(f'images/{i}')
        
        img = pred[i].reshape((28,28))

        plt.imshow(img,cmap='gray')
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
        plt.savefig(f'images/{i}/epoch_{epoch}.png',dpi=200)
        plt.cla()
        plt.close()
