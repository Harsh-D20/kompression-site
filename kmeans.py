import numpy as np


def cluster_to_img_distance(cl, img):
    """
    Calculates squared Euclidean distance from a single cluster center to all pixels in an image.

    Args:
        cl (3,): RGB value of the cluster
        img (R, C, 3): R x C array with RGB values

    Returns:
        (R, C): array of squared distances
    """
    out = np.subtract(img, cl)
    out = np.sum(out**2, axis=2)
    return np.array(out)


def assign_clusters(cls, img):
    """Assigns each pixel in the image to the closest cluster center.

    Args:
        cls (N, 3): Array of RGB values that are cluster centers
        img (R, C, 3): R x C array with RGB values

    Returns:
        (R, C): Array with each entry [i,j] being the cluster (0 to K-1) assigned to that pixel
    """
    h, w, c = img.shape
    pixels_flat = img.reshape(-1, c)  # N = R * C

    # Convert list of clusters to a NumPy array (K, 3)
    centers = np.array(cls, dtype=float)

    # pixels_flat is (N, 3), centroids is (K, 3)
    # pixels_flat[:, np.newaxis, :] becomes (N, 1, 3)
    # centers[np.newaxis, :, :] becomes (1, K, 3)
    # (N, 1, 3) - (1, K, 3) broadcasts to (N, K, 3)
    # Summing over axis=2 gives (N, K) -> distances from each pixel to each center
    squared_distances = np.sum(
        (pixels_flat[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2
    )

    # Find the index of the minimum distance for each pixel
    assigned_img = np.argmin(squared_distances, axis=1)

    # Reshape labels back to image dimensions (R, C)
    return assigned_img.reshape(h, w)


def update_centers(asgn, img, k):
    """Updates the cluster centers based on the mean of assigned pixels.

    Args:
        asgn (R, C): Array of each pixel's assigned cluster
        img (R, C, 3): R x C array with RGB values
        k (int): number of clusters

    Returns:
        _type_: _description_
    """
    centers = [np.array([-255, -255, -255])] * k
    count = [0.0] * k
    img = np.array(img, np.float32)
    # for each pixel, add up the BGR values for each cluster
    for r in range(0, img.shape[0]):
        for c in range(0, img.shape[1]):
            # get cluster pixel is assigned to
            cluster = asgn[r][c]
            # if that cluster has no pixels
            if count[cluster] == 0:
                centers[cluster] = img[r][c]
            # add pixel value to cluster center
            else:
                centers[cluster] = np.add(centers[cluster], img[r][c])
            count[cluster] += 1
    # divide each cluster center by the number of values for the cluster
    # if number of values is 0, set center to (-255, -255, -255)
    for i in range(0, len(centers)):
        if count[i] == 0:
            centers[i] = np.array([-255, -255, -255])
        else:
            centers[i] = np.clip(np.divide(centers[i], count[i]), 0, 255)
    return centers


def initialized_k_means(cls, img, n):
    old_asgn = None
    for _ in range(0, n):
        # assign every pixel to a cluster
        asgn = assign_clusters(cls, img)
        # if the algorithm has converged, break
        if old_asgn is not None and np.array_equal(old_asgn, asgn):
            break
        # update the clusters
        cls = update_centers(asgn, img, len(cls))
        old_asgn = asgn.copy()
    return cls, asgn


def quantize_image(img, k, n):
    r, c, _ = img.shape
    # create a random cluster and add it to cls
    cls = []
    cls.append(np.random.randint(0, 255, 3))
    for _ in range(2, k + 1):
        # create a new array with all values set to inf
        min_dsts = np.full((r, c), np.inf)
        # for each cluster, get the distance for that cl to all pixels in img and update min_dsts
        for cl in cls:
            dst = cluster_to_img_distance(cl, img)
            min_dsts = np.minimum(min_dsts, dst)
        # get the total distance
        total_dst = np.sum(min_dsts)
        # for each dst, divide by total dst. If min_dst is low, then probability
        # of the new cluster being that pixel is low. If min_dst is high, then
        # probability of the new cluster being that pixel is high
        probs = min_dsts / total_dst if total_dst > 0 else min_dsts
        # randomly sample from the probability distribution and add the sampled cluster to cls
        idx = np.random.choice(r * c, p=probs.ravel())
        cls.append(img[np.unravel_index(idx, (r, c))])

    pixel_values = [[[0.0, 0.0, 0.0]] * img.shape[1] for _ in range(img.shape[0])]
    cls, asgn = initialized_k_means(cls, img, n)
    for r in range(0, img.shape[0]):
        for c in range(0, img.shape[1]):
            pixel_values[r][c] = cls[asgn[r][c]]
    out_img = np.array(pixel_values, np.uint8)
    return out_img, cls
