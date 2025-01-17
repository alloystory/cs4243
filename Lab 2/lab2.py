""" CS4243 Lab 2: Image Segmentations
See accompanying Jupyter notebook (lab2.ipynb) and PDF (lab2.pdf) for instructions.
"""
import cv2
import numpy as np
import random

from time import time
from skimage import color
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed


# Part 1 

def smoothing(img):
    """Smooth image using Guassain filter.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).

    Returns:
        img_smoothed (np.ndarray)   : Output smoothed image of size (H, W, 3).

    """

    """ YOUR CODE STARTS HERE """
    img_smoothed = cv2.GaussianBlur(img, (5, 5), 5)
    """ YOUR CODE ENDS HERE """

    return img_smoothed

def RGBtoLab(img):
    """Convert RGB image into L*a*b color space.

    Args:
        img (np.ndarray)            : Input RGB image  of size (H, W, 3).


    Returns:
        lab (np.ndarray)            : Converted L*a*b image of size (H, W, 3).

    """

    """ YOUR CODE STARTS HERE """
    lab = color.rgb2lab(img)
    """ YOUR CODE ENDS HERE """
   
    return lab



# Part 2
def k_means_clustering(data,k):
    """ Estimate clustering centers using k-means algorithm.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        k (int)                     : Number of centroids

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """
    start = time()

    """ YOUR CODE STARTS HERE """
    
    # Randomly pick k centers
    centers = []
    for _ in range(k):
        i = random.randint(0, len(data) - 1)
        centers.append(data[i])
    centers = np.array(centers)
    
    iter_limit = 50
    threshold_per_center = 1e-4
    labels = np.zeros(len(data), dtype = int)
    
    for _ in range(iter_limit):
        
        # Assign each point in data to a center
        for i in range(len(data)):
            p_i = data[i]
            y_i = np.argmin(np.sum(np.square(p_i - centers), axis = 1))
            labels[i] = y_i

        # Calculate new center based on labelling
        iter_distance = 0
        for i in range(k):
            data_i = data[labels == i]
            new_center = np.average(data_i, axis = 0)
            distance = np.sqrt(np.sum(np.square(new_center - centers[i])))
            iter_distance += distance
            centers[i] = new_center
        
        # Terminate if centers move less than threshold
        if iter_distance < threshold_per_center * k:
            break

    """ YOUR CODE ENDS HERE """


    end =  time()
    kmeans_runtime = end - start
    print("K-means running time: %.3fs."% kmeans_runtime)
    return labels, centers



def get_bin_seeds(data, bin_size, min_bin_freq=1):
    """ Generate initial bin seeds for windows sampling.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        bin_size (float)            : Bandwidth.
        min_bin_freq (int)          : For each bin_seed, number of the minimal points should cover.

    Returns:
        bin_seeds (List)            : Reprojected bin seeds. All bin seeds with total point number 
                                      bigger than the threshold.
    """

    """ YOUR CODE STARTS HERE """
    
    bin_seeds = []
    compressed = np.round(data / bin_size)
    seeds, counts = np.unique(compressed, axis=0, return_counts=True)
    reprojected = seeds * bin_size
    for seed, count in zip(reprojected, counts):
        if count >= min_bin_freq:
            bin_seeds.append(seed)
    
    """ YOUR CODE ENDS HERE """
    return bin_seeds

def mean_shift_single_seed(start_seed, data, nbrs, max_iter):
    """ Find mean-shift peak for given starting point.

    Args:
        start_seed (np.ndarray)     : Coordinate (x, y) of start seed. 
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        nbrs (class)                : Class sklearn.neighbors._unsupervised.NearestNeighbors.
        max_iter (int)              : Max iteration for mean shift.

    Returns:
        peak (tuple)                : Coordinate (x,y) of peak(center) of the attraction basin.
        n_points (int)              : Number of points in the attraction basin.
                              
    """

    # For each seed, climb gradient until convergence or max_iter
    bandwidth = nbrs.get_params()['radius']
    stop_thresh = 1e-3 * bandwidth  # when mean has converged

    """ YOUR CODE STARTS HERE """
    
    peak, neighbors = start_seed, None
    neigh = NearestNeighbors(radius=bandwidth)
    neigh.fit(data)

    for _ in range(max_iter):
        indices = neigh.radius_neighbors([peak], return_distance=False)[0]
        neighbors = data[indices]

        new_peak = np.average(neighbors, axis = 0)
        distance = np.sqrt(np.sum(np.square(peak - new_peak)))
        
        peak = new_peak

        if distance < stop_thresh:
            break

    peak = tuple(peak)
    n_points = len(neighbors)

    """ YOUR CODE ENDS HERE """

    return peak, n_points


def mean_shift_clustering(data, bandwidth=0.7, min_bin_freq=5, max_iter=300):
    """pipline of mean shift clustering.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        bandwidth (float)           : Bandwidth parameter for mean shift algorithm.
        min_bin_freq(int)           : Parameter for get_bin_seeds function.
                                      For each bin_seed, number of the minimal points should cover.
        max_iter (int)              : Max iteration for mean shift.

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)
    """
    start = time()
    n_jobs = None
    seeds = get_bin_seeds(data, bandwidth, min_bin_freq)
    n_samples, n_features = data.shape
    center_intensity_dict = {}

    # We use n_jobs=1 because this will be used in nested calls under
    # parallel calls to _mean_shift_single_seed so there is no need for
    # for further parallelism.
    nbrs = NearestNeighbors(radius=bandwidth, n_jobs=1).fit(data)
    # execute iterations on all seeds in parallel
    all_res = Parallel(n_jobs=n_jobs)(
        delayed(mean_shift_single_seed)
        (seed, data, nbrs, max_iter) for seed in seeds)

    # copy results in a dictionary
    for i in range(len(seeds)):
        if all_res[i] is not None:
            center_intensity_dict[all_res[i][0]] = all_res[i][1]

    if not center_intensity_dict:
        # nothing near seeds
        raise ValueError("No point was within bandwidth=%f of any seed."
                         " Try a different seeding strategy \
                         or increase the bandwidth."
                         % bandwidth)
    


    """ YOUR CODE STARTS HERE """

    removed_centers = set()
    potential_centers = np.array(list(center_intensity_dict.keys()))

    neigh = NearestNeighbors(radius=bandwidth)
    neigh.fit(potential_centers)

    for center in potential_centers:
        ### find neighbor centers within bandwidth
        indices = neigh.radius_neighbors([center], return_distance=False)[0]
        neighbors = potential_centers[indices]

        ### find the center with highest count
        count_to_centers = {center_intensity_dict[tuple(center)]: tuple(center) for center in neighbors}
        center_w_largest_count = count_to_centers[max(count_to_centers)]

        ### add the rest into removed_centers 
        for center in neighbors:
            if tuple(center) != center_w_largest_count:
                removed_centers.add(tuple(center))

    ### remove centers that are in removed_centers from potential_centers
    centers = np.array([center for center in potential_centers if tuple(center) not in removed_centers])

    labels = np.array([np.argmin(np.sum(np.square(p_i - centers), axis = 1)) for p_i in data])

    """ YOUR CODE ENDS HERE """
    end =  time()
    kmeans_runtime = end - start
    print("mean shift running time: %.3fs."% kmeans_runtime)
    return labels, centers









#Part 3:

def k_means_segmentation(img, k):
    """Descrption.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).
        k (int)                     : Number of centroids
    
    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """

    """ YOUR CODE STARTS HERE """
    img = img / 255.
    
    if len(img.shape) == 2:
        H, W = img.shape
        data = img.reshape(H*W, 1)
    else:
        H, W, C = img.shape
        data = img.reshape(H*W, C)
    
    labels, centers = k_means_clustering(data, k)
    """ YOUR CODE ENDS HERE """

    return labels,centers


def mean_shift_segmentation(img,b):
    """Descrption.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).
        b (float)                     : Bandwidth.

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """

    """ YOUR CODE STARTS HERE """
    
    if len(img.shape) == 2:
        H, W = img.shape
        data = img.reshape(H*W, 1)
    else:
        H, W, C = img.shape
        data = img.reshape(H*W, C)
    
    labels, centers = mean_shift_clustering(data=data, bandwidth=b)
    
    """ YOUR CODE ENDS HERE """

    return labels, centers














"""Helper functions: You should not have to touch the following functions.
"""
def load_image(im_path):
    """Loads image and converts to RGB format

    Args:
        im_path (str): Path to image

    Returns:
        im (np.ndarray): Loaded image (H, W, 3), of type np.uint8.
    """


    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def colors(k):

    """generate the color for the plt.scatter.

    Args:
        k (int): the number of the centroids

    Returns:
        ret (list): list of colors .

    """

    colour = ["coral", "dodgerblue", "limegreen", "deeppink", "orange", "darkcyan", "rosybrown", "lightskyblue", "navy"]
    if k <= len(colour):
        ret = colour[0:k]
    else:
        ret = []
        for i in range(k):
            ret.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
    return ret

def stack_seg(img, labels, centers):
    """stack segmentations for visualization.

    Args:
        img (np.ndarray): image
        labels(np.ndarray): lables for every pixel. 
        centers(np.ndarray): cluster centers.

    Returns:
        np.vstack(result) (np.ndarray): stacked result.

    """

    labels = labels.reshape((img.shape[:-1]))
    reduced = np.uint8(centers)[labels]
    result = [np.hstack([img])]
    for i, c in enumerate(centers):
        mask = cv2.inRange(labels, i, i)
        mask = np.dstack([mask]*3) # Make it 3 channel
        ex_img = cv2.bitwise_and(img, mask)
        result.append(np.hstack([ex_img]))

    return np.vstack(result)
