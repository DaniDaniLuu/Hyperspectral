import numpy as np
import matplotlib.pyplot as plt
from bioio import BioImage 
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA 
from scipy.ndimage import gaussian_filter

plt.ion()

czi_path = "241121_10h40min_bra.h2b.mAp_twist.RFP_crbn.h2b.GFP_meis.kaede.czi"
img = BioImage(czi_path)

print("Image metadata:")
print("dims X", img.dims.X)
print("Scenes:", img.scenes)

data_5d = img.data  
print("Loaded image shape (T, C, Z, Y, X):", data_5d.shape)

data_cube = data_5d[0, :, 0, :, :]  # shape: (C, Y, X)
# The data cube = 3d matrix for the intensity of channel c at pixel (y, x)
print("Spectral cube shape (Channels, Height, Width):", data_cube.shape)

# Gaussian filter to each channel
# Smoothed_data_cube = gaussian_filter(data_cube, sigma=(0,1,1))


# Dimensions of the cube (C = channels, H = height, W = width)
C, H, W = data_cube.shape
num_pixels = H * W

print("Pixel count", num_pixels)

# Reshaping the data cube into 2d matrix 
spectral_matrix = data_cube.reshape(C, num_pixels).T # Transposing matrix into (num_pixels, C) to have each row correspond to a single pixel
print("Spectral matrix shape (Pixels, Channels):", spectral_matrix.shape)

# Creating a PCA Object
pca = PCA(n_components=4)

# Fit PCA onto data and transforms it.
spectra_reduced = pca.fit_transform(spectral_matrix)

# spectra_reduced now has shape (n_pixels, n_components)
print("Reduced data shape:", spectra_reduced.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Plotting PCA results
plt.figure(figsize=(8, 6))
plt.scatter(spectra_reduced[:, 0], spectra_reduced[:, 1], s=1, alpha=0.3)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Hyperspectral Data")
plt.show()



# K-Means clustering
R = 3  # number of expected endmembers

# Run K-Means on the spectral matrix
kmeans = KMeans(n_clusters=R, random_state=42)
labels = kmeans.fit_predict(spectral_matrix)
cluster_centers = kmeans.cluster_centers_  # shape: (R, C)

print("Derived %d cluster centroids (endmember candidates) of length %d:" % (R, cluster_centers.shape[1]))
for j, centroid in enumerate(cluster_centers, 1):
    print(f"Endmember {j} centroid spectrum:", centroid)
    

# Plotting K-means clustering results
plt.figure(figsize=(10, 8))
# Scatter plot of points colored by their clusters
plt.scatter(spectra_reduced[:, 0], spectra_reduced[:, 1], 
        c=labels, cmap='viridis', 
        s=1, alpha=0.6)

# Plot cluster centers
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
        c='red', marker='x', s=200, linewidths=3, 
        label='Cluster Centers')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering Results (k={})'.format(R))
plt.legend()
plt.colorbar(label='Cluster Label')
plt.draw()
plt.pause(0.1)

# Creating a visualization of the spatial distribution of clusters
cluster_image = labels.reshape(H, W)
plt.figure(figsize=(10, 8))
plt.imshow(cluster_image, cmap='viridis')
plt.title('Spatial Distribution of Clusters')
plt.colorbar(label='Cluster Label')
plt.axis('off')
plt.draw()
plt.pause(0.1)

# Plotting spectral signatures of selected pixels
sample_pixels = [(591, 357)]
for (y, x) in sample_pixels:
    spectrum = data_cube[:, y, x]
    plt.plot(range(1, C+1), spectrum, marker='o', label=f"Pixel ({y},{x})")
plt.xlabel("Spectral Channel Index")
plt.ylabel("Intensity (a.u.)")
plt.title("Spectral signatures of selected pixels")
plt.legend()
plt.draw()
plt.pause(0.1)

plt.ioff()
plt.show(block=True)