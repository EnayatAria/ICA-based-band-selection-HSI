'''
This code is an algorithm for band selection of hyperspectral images
using ICA (independent component analysis) method.
Reference: "Band Selection Using Independent Component Analysis for Hyperspectral
Image Processing", Hongtao Du, et al. 2003

Inputs:
        Should be given:
            A hyperspectral data-set in ENVI format
        Will be asked via the console:
            Number of components
            Number of the bands required to be selected


The number of component is determined by the user
It can be evaluated to minimise the difference
between the original band set (X) and multiplication of mixing matrix (A_) and the source (S_)

This evaluation can be checked by the 'assert' function and varying the 'atol' and 'rtol' parameters (line 62)
The smaller values of the parameters that pass the 'assert' test function, give the better estimation of the mixing matrix
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA

# to read ENVI format images
import spectral.io.envi as envi


###########################################################

# Reading a hyperspectral image data set in envi format
img = envi.open('190_bands.hdr', '190_bands').load()   # load Salinas scene
# vectorize the image
X=np.reshape(img, (img.shape[0]*img.shape[1],img.shape[2]))

# free the occupied memory
del img


# compute ICA
# Get number of components from user

n_c=input('\033[1m' + '\033[93m'+"What is the number of components? ")
if int(n_c)>X.shape[1]:
    print('\033[31m'+"Error: The Number of components is higher than the number of bands in the data set")
    exit()

n_b=input("How many bands do you want to select? ")
if int(n_b)>X.shape[1]:
    print('\033[31m'+"Warning: The Number of bands supposed to be selected is higher than the number of bands in the data set"+'\033[0m')


# Whitening should be done for pre-processing of the data
ica=FastICA(n_components=int(n_c), whiten=True)     #Indian Pine with 160 components has the best estimation of the mixing matrix
S_ = ica.fit_transform(X)        # Reconstruct signals
A_=ica.mixing_                 # Get estimated mixing matrix

# To check the unmixing matrix, we can use the following line
    #assert np.allclose(X, np.dot(S_,A_.T) + ica.mean_,atol=0.0001,rtol=0.13)

# compute the rank of a matrix for non-square matrix
if A_.shape[1] != np.linalg.matrix_rank(A_):
    print('A does not have left inverse ')
else:
    # compute the pseudo-inverse of A_
    W=np.linalg.pinv(A_)    # W. transpose(X)=transpose(S_)
    assert np.allclose(A_, np.dot(A_, np.dot(W, A_)))  # to check the pseudo-inverse matrix
B_W=np.sum(np.absolute(W),axis=0)   # compute a weight per band
sortB_W =np.argsort(B_W)   # extract the indexes
print('\033[94m'+'Band number with the highest scores : '+str(sortB_W[-int(n_b):]+1)+'\033[0m')    # to get the last elements of the list having higher weight

