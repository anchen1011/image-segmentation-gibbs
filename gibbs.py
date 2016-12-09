from __future__ import division
import itertools
import sys
import random

import numpy as np
from numpy import newaxis as nax
import scipy.misc as misc

import matplotlib.pyplot as plt

ALPHA = 0.9

# These two are hardcoded, but could be estimated using maximum likelihood
# from parts of the image that are labeled.
SIGMA = 40.0
MUS = np.array([[112, 85, 56], [174, 187, 187]])

def gibbs_segmentation(image, burn_in, collect_frequency, n_samples):
    """
    Uses Gibbs sampling to segment an image into foreground and background.

    Inputs
    ------
    image : a numpy array containing the image. Should be Nx x Ny x 3
    burn_in : Number of iterations to run as 'burn-in' before collecting data
    collect_frequency : How many samples in between collected samples
    n_samples : how many samples to collect in total

    Returns
    -------
    A distribution of the collected samples: a numpy array with a value between
    0 and 1 (inclusive) at every pixel.
    """
    (Nx, Ny, _) = image.shape

    distribution = np.zeros( (Nx, Ny) )

    # Initialize binary estimates at every pixel randomly. 
    estimates = (np.random.random( (Nx, Ny) ) > .5).astype(int)

    total_iterations = burn_in + (collect_frequency * (n_samples - 1) + 1)
    pixel_indices = list(itertools.product(range(Nx),range(Ny)))

    for iteration in range(total_iterations):

        # Loop over entire grid, using a random order for faster convergence
        random.shuffle(pixel_indices)
        for (i,j) in pixel_indices:
            xf = observation_model(image[i,j,:], 0)
            xb = observation_model(image[i,j,:], 1)
            for neighbor in get_neighbors(estimates, i, j):
                xf *= edge_model(0, neighbor)
                xb *= edge_model(1, neighbor)
            pb = xb / (xf + xb)
            estimates[i,j] = (np.random.random() < pb).astype(int)
        if iteration > burn_in and (iteration - burn_in + collect_frequency)%collect_frequency == 1:
            distribution += estimates
    
    distribution /= n_samples

    return distribution

def get_neighbors(grid, x, y):
    """
    Returns values of the grid at points neighboring (x,y)

    Inputs
    ------
    grid   : a 2d numpy array
    (x, y) : valid indices into the grid

    Returns
    -------
    An array of values neighboring (x,y) in grid. Returns
    a 2-element array if (x,y) is at a corner,
    a 3-element array if (x,y) is at an edge, and
    a 4-element array if (x,y) is not touching an edge/corner
    """
    out = []
    if x > 0:
        out.append(grid[x-1, y])
    if y > 0:
        out.append(grid[x, y-1])
    if y < grid.shape[1] - 1:
        out.append(grid[x, y+1])
    if x < grid.shape[0] - 1:
        out.append(grid[x+1, y])
    return out

def compute_observation_model():

    all_intensities = np.arange(256) # 8-bit pixel values can be 0, ... , 255
    dist = np.exp(-((all_intensities[nax, nax, :] - MUS[:,:,nax]) / SIGMA) ** 2)
    normalization = np.sum(dist, 2)
    dist /= normalization[:,:,nax]
    assert np.allclose(np.sum(dist,2) , 1)

    def observation_model(intensities, label):
        """
        Computes the probability of observing a set of RGB intensities for a
        given label.

        Inputs
        ------
        intensities : a 3-element array of integers from 0 to 255 with RGB
                      intensities
        label : either 0 (foreground) or 1 (background)

        Returns
        -------
        The probability of observing these intensities from that label
        """
        idxs = enumerate(intensities)
        return np.prod([dist[label, i, intensity] for (i, intensity) in idxs])

    return observation_model

# Create the observation_model function
observation_model = compute_observation_model()

def edge_model(label1, label2):
    """
    Given the values at two pixels, returns the edge potential between
    those two pixels.

    Hint: you may not need to use this function at all - there might be a more
    efficient way to compute this for an array of values using numpy!

    Inputs
    ------
    label1 : either 0 (foreground) or 1 (background)
    label2 : either 0 (foreground) or 1 (background)

    Returns
    -------
    The edge potential between two pixels with values label1 and label2.
    """
    if label1 == label2:
        return ALPHA
    else:
        return 1-ALPHA


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <image file>' % sys.argv[0])
        sys.exit(1)

    image_file = sys.argv[1]

    image = misc.imread(image_file)
    seg_image = gibbs_segmentation(image, 5, 2, 100)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes.flat[0].imshow(image)
    plt.gray()
    im = axes.flat[1].imshow(seg_image.astype('float64'), interpolation='nearest')
    fig.colorbar(im, ax=axes.ravel().tolist())
    print('Close displayed figure to terminate...')
    plt.show()
