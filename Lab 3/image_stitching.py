import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve
import math

from utils import pad, get_output_space, unpad

import cv2
_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame


def warp_image(src, dst, h_matrix):
    dst = dst.copy()
    dst = cv2.warpPerspective(dst, np.linalg.inv(h_matrix), (src.shape[1] + dst.shape[1], src.shape[0]))
    dst[0:src.shape[0], 0:src.shape[1]] = src
    return dst

def draw_matches(im1, im2, im1_pts, im2_pts, inlier_mask=None):
    """Generates a image line correspondences

    Args:
        im1 (np.ndarray): Image 1
        im2 (np.ndarray): Image 2
        im1_pts (np.ndarray): Nx2 array containing points in image 1
        im2_pts (np.ndarray): Nx2 array containing corresponding points in
          image 2
        inlier_mask (np.ndarray): If provided, inlier correspondences marked
          with True will be drawn in green, others will be in red.

    Returns:

    """
    height1, width1 = im1.shape[:2]
    height2, width2 = im2.shape[:2]
    canvas_height = max(height1, height2)
    canvas_width = width1 + width2

    canvas = np.zeros((canvas_height, canvas_width, 3), im1.dtype)
    canvas[:height1, :width1, :] = im1
    canvas[:height2, width1:width1+width2, :] = im2

    im2_pts_adj = im2_pts.copy()
    im2_pts_adj[:, 0] += width1

    if inlier_mask is None:
        inlier_mask = np.ones(im1_pts.shape[0], dtype=np.bool)

    # Converts all to integer for plotting
    im1_pts = im1_pts.astype(np.int32)
    im2_pts_adj = im2_pts_adj.astype(np.int32)

    # Draw points
    all_pts = np.concatenate([im1_pts, im2_pts_adj], axis=0)
    for pt in all_pts:
        cv2.circle(canvas, (pt[0], pt[1]), 4, _COLOR_BLUE, 2)

    # Draw lines
    for i in range(im1_pts.shape[0]):
        pt1 = tuple(im1_pts[i, :])
        pt2 = tuple(im2_pts_adj[i, :])
        color = _COLOR_GREEN if inlier_mask[i] else _COLOR_RED
        cv2.line(canvas, pt1, pt2, color, 2)

    return canvas

def transform_homography(src, h_matrix, getNormalized = True):
    """Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    """
    transformed = None

    input_pts = np.insert(src, 2, values=1, axis=1)
    transformed = np.zeros_like(input_pts)
    transformed = h_matrix.dot(input_pts.transpose())
    if getNormalized:
        transformed = transformed[:-1]/transformed[-1]
    transformed = transformed.transpose().astype(np.float32)
    
    return transformed

def compute_homography(src, dst):
    """Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    """
    h_matrix = np.eye(3, dtype=np.float64)

    ### YOUR CODE HERE
    N = src.shape[0]
    h_src = pad(src)
    h_dst = pad(dst)
    
    # Finding the transformation T that normalises the points
    # Compute based on link given in lecture slides
    m_src = np.mean(h_src, axis = 0)
    s_src = np.sqrt(2) / np.std(h_src)
    m_dst = np.mean(h_dst, axis = 0)
    s_dst = np.sqrt(2) / np.std(h_dst)
    
    T_src = np.array([
        [s_src,  0,       -s_src*m_src[0]],
        [0,      s_src,   -s_src*m_src[1]],
        [0,      0,       1]
    ])
    T_dst = np.array([
        [s_dst,  0,       -s_dst*m_dst[0]],
        [0,      s_dst,   -s_dst*m_dst[1]],
        [0,      0,       1]
    ])
    
    norm_src = np.dot(T_src, h_src.transpose()).transpose()
    norm_dst = np.dot(T_dst, h_dst.transpose()).transpose()
    
    # Standard DLT
    # Setting up matrix A
    A = np.zeros((2*N, 9))
    for i in range(N):
        x_src, y_src = norm_src[i][0], norm_src[i][1]
        x_dst, y_dst = norm_dst[i][0], norm_dst[i][1]
        A[2*i] = [-x_src, -y_src, -1, 0, 0, 0, x_src*x_dst, y_src*x_dst, x_dst]
        A[2*i+1] = [0, 0, 0, -x_src, -y_src, -1, x_src*y_dst, y_src*y_dst, y_dst]
    
    # Performing SVD on A
    u, s, vh = np.linalg.svd(A, full_matrices = True)
    H = vh[-1].reshape((3, 3))
    
    # Denormalising data
    h_matrix = np.linalg.multi_dot([np.linalg.inv(T_dst), H, T_src])
    
    # Making H affine
    h_matrix = h_matrix / h_matrix[2, 2]
    ### END YOUR CODE

    return h_matrix

def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the functions filters.sobel_v filters.sobel_h & scipy.ndimage.filters.convolve, 
        which are already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    ### YOUR CODE HERE
    # Finding image gradients
    I_x = filters.sobel_v(img)
    I_y = filters.sobel_h(img)
    
    I_x2 = I_x ** 2
    I_y2 = I_y ** 2
    I_xy = I_x * I_y
    
    # Finding sum of gradients in each window
    A = convolve(I_x2, window)
    B = convolve(I_xy, window)
    C = convolve(I_y2, window)
    
    # Finding response
    for i in range(H):
        for j in range(W):
            M = np.array([
                [A[i, j], B[i, j]],
                [B[i, j], C[i, j]]
            ])
            det = np.linalg.det(M)
            trace = np.trace(M)
            response[i, j] = det - (k * (trace ** 2))
    ### END YOUR CODE

    return response

def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.
    
    Hint:
        If a denominator is zero, divide by 1 instead.
    
    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    """
    feature = []
    ### YOUR CODE HERE
    h, w = patch.shape
    feature = patch.reshape(h * w)
    sigma = feature.std()
    mean = feature.mean()
    feature = feature - mean
    if sigma > 0:
        feature = feature / sigma
    ### END YOUR CODE
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)

def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed 
    when the distance to the closest vector is much smaller than the distance to the 
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.
    
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints
        
    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair 
        of matching descriptors
    """
    matches = []
    
    N = desc1.shape[0]
    dists = cdist(desc1, desc2)
    printed = False

    ### YOUR CODE HERE
    
    for i in range(N):
        dist_1_2 = dists[i]
        
        # Distance to the closest vector
        a_j = dist_1_2.argmin()
        a = dist_1_2[a_j]
        dist_1_2[a_j] = float("inf")
        
        # Distance to the second-closest vector
        b_j = dist_1_2.argmin()
        b = dist_1_2[b_j]
        
        ratio = a / b
        if ratio < threshold:
            matches.append([i, a_j])
    matches = np.array(matches)
    
    ### END YOUR CODE
    
    return matches

def ransac(keypoints1, keypoints2, matches, sampling_ratio=0.5, n_iters=500, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * sampling_ratio)

    # Please note that coordinates are in the format (y, x)
    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])
    matched1_unpad = keypoints1[matches[:,0]]
    matched2_unpad = keypoints2[matches[:,1]]

    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    ### YOUR CODE HERE    
    
    # Flip x and y coordinates
    matched1[:, [0,1]] = matched1[:, [1,0]]
    matched2[:, [0,1]] = matched2[:, [1,0]]
    matched1_unpad[:, [0,1]] = matched1_unpad[:, [1,0]]
    matched2_unpad[:, [0,1]] = matched2_unpad[:, [1,0]]
    
    # Ransac Loop
    for _ in range(n_iters):
        # Randomly pick n_samples
        rand_indices = np.random.choice(N, n_samples)
        rand_1 = matched1_unpad[rand_indices]
        rand_2 = matched2_unpad[rand_indices]
        
        # Compute H using these correspondences
        iter_H = compute_homography(rand_1, rand_2)
        
        # Count the number of inliers
        iter_n_inliers = 0
        iter_inliers = np.zeros(N)
        for i in range(N):
            p1 = matched1_unpad[i]
            p2 = matched2[i]
            p2_prime = np.dot(np.linalg.inv(iter_H), p2.transpose()).transpose()
            p2_prime = p2_prime[:2] / p2_prime[-1]
            dist = np.sum(np.square(p1 - p2_prime))
            
            if dist < threshold:
                iter_n_inliers += 1
                iter_inliers[i] = 1

        # Store the max number of inliers
        if iter_n_inliers > n_inliers:
            n_inliers = iter_n_inliers
            max_inliers = iter_inliers
    
    # Recomputing H with all of the max inliers
    max_inliers = max_inliers.astype(bool)
    max_src = matched1_unpad[max_inliers]
    max_dst = matched2_unpad[max_inliers]
    H = compute_homography(max_src, max_dst)
    ### END YOUR CODE
    
    return H, matches[max_inliers]


def sift_descriptor(patch):
    """
    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each length of 16/4=4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
        patch: grayscale image patch of shape (h, w)

    Returns:
        feature: 1D array of shape (128, )
    """
    
    dx = filters.sobel_v(patch)
    dy = filters.sobel_h(patch)
    histogram = np.zeros((4,4,8))
    
    ### YOUR CODE HERE
    # get gradient magnitudes and orientation of each pixel
    # using the formulas in lab pdf
    m = np.sqrt(dx ** 2 + dy ** 2)
    t = np.arctan2(dy, dx)  # Array of angles in radians, in the range [-pi, pi].

    for r in range(0, 16, 4):
        for c in range(0, 16, 4):
            magnitude = m[r:r+4, c:c+4]
            theta = t[r:r+4, c:c+4]

            # get histogram of local distribution of gradient magnitudes in 8 orientations
            local_histogram, _ = np.histogram(theta, bins=8, range=(-np.pi, np.pi),
                                              weights=magnitude)

            # normalize histogram
            local_histogram = local_histogram / np.sum(local_histogram)

            histogram[r//4][c//4] = local_histogram

    feature = histogram.reshape(128)
    # END YOUR CODE
    
    return feature


def linear_blend(img1_warped, img2_warped):
    """
    Linearly blend img1_warped and img2_warped by following the steps:

    1. Define left and right margins (already done for you)
    2. Define a weight matrices for img1_warped and img2_warped
        np.linspace and np.tile functions will be useful
    3. Apply the weight matrices to their corresponding images
    4. Combine the images

    Args:
        img1_warped: Refernce image warped into output space
        img2_warped: Transformed image warped into output space

    Returns:
        merged: Merged image in output space
    """
    out_H, out_W = img1_warped.shape # Height and width of output space
    img1_mask = (img1_warped != 0)  # Mask == 1 inside the image
    img2_mask = (img2_warped != 0)  # Mask == 1 inside the image

    # Find column of middle row where warped image 1 ends
    # This is where to end weight mask for warped image 1
    right_margin = out_W - np.argmax(np.fliplr(img1_mask)[out_H//2, :].reshape(1, out_W), 1)[0]

    # Find column of middle row where warped image 2 starts
    # This is where to start weight mask for warped image 2
    left_margin = np.argmax(img2_mask[out_H//2, :].reshape(1, out_W), 1)[0]

    ### YOUR CODE HERE
    raise NotImplementedError() # Delete this line
    ### END YOUR CODE

    return merged


def stitch_multiple_images(imgs, desc_func=simple_descriptor, patch_size=5):
    """
    Stitch an ordered chain of images together.

    Args:
        imgs: List of length m containing the ordered chain of m images
        desc_func: Function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: Size of square patch at each keypoint

    Returns:
        panorama: Final panorma image in coordinate frame of reference image
    """
    # Detect keypoints in each image
    keypoints = []  # keypoints[i] corresponds to imgs[i]
    for img in imgs:
        kypnts = corner_peaks(harris_corners(img, window_size=3),
                              threshold_rel=0.05,
                              exclude_border=8)
        keypoints.append(kypnts)
    # Describe keypoints
    descriptors = []  # descriptors[i] corresponds to keypoints[i]
    for i, kypnts in enumerate(keypoints):
        desc = describe_keypoints(imgs[i], kypnts,
                                  desc_func=desc_func,
                                  patch_size=patch_size)
        descriptors.append(desc)
    # Match keypoints in neighboring images
    matches = []  # matches[i] corresponds to matches between
                  # descriptors[i] and descriptors[i+1]
    for i in range(len(imgs)-1):
        mtchs = match_descriptors(descriptors[i], descriptors[i+1], 0.7)
        matches.append(mtchs)

    ### YOUR CODE HERE
    raise NotImplementedError() # Delete this line
    ### END YOUR CODE

    return panorama
