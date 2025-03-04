#%%

# Load all libraries used
import matplotlib.pyplot as plt
import copy
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans
from matplotlib.image import imread

def initialize_clusters(points: np.ndarray, k_clusters: int) -> np.ndarray:
    #Initializes and returns k random centroids from the given dataset.
    num_points = points.shape[0]
    random_indices = np.random.choice(num_points, k_clusters, replace=False)
    initial_clusters = points[random_indices]
    return initial_clusters

def calculate_metric(points: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    #Computes the Euclidean distance between each point and a given centroid.
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = np.linalg.norm(points[i] - centroid)
    return distances

def compute_distances(points: np.ndarray, centroids_points: np.ndarray) -> np.ndarray:
    # Computes and returns the distance from each point to each centroid.
    k_clusters = centroids_points.shape[0]
    num_points = points.shape[0]
    distances_array = np.zeros((k_clusters, num_points))
    for i in range(k_clusters):
        distances_array[i] = calculate_metric(points, centroids_points[i])
    return distances_array

def assign_centroids(distances: np.ndarray) -> np.ndarray:
    # Assigns each point to the closest centroid based on the distances.
    num_points = distances.shape[1]
    assigned_centroids = np.zeros(num_points, dtype=np.int32)
    for i in range(num_points):
        assigned_centroids[i] = np.argmin(distances[:, i])
    return assigned_centroids

def calculate_objective(assigned_centroids: np.ndarray, distances: np.ndarray) -> np.float32:
    # Calculates and returns the objective function value for the clustering.
    num_points = assigned_centroids.shape[0]
    total_distance = 0
    for i in range(num_points):
        cluster_idx = assigned_centroids[i]
        total_distance += distances[cluster_idx, i]
    return np.float32(total_distance)

def calculate_new_centroids(points: np.ndarray, assigned_centroids: np.ndarray, k_clusters: int) -> np.ndarray:
    # Computes new centroids based on the current cluster assignments.
    new_clusters = np.zeros((k_clusters, points.shape[1]), dtype=np.float32)
    for k in range(k_clusters):
        cluster_points = points[assigned_centroids == k]
        if cluster_points.shape[0] > 0:
            new_clusters[k] = np.mean(cluster_points, axis=0)
    return new_clusters

def fit(points: np.ndarray, k_clusters: int, n_of_iterations: int, error: float = 0.001) -> tuple:
    # Fits the k-means clustering model on the dataset.
    centroid_points = initialize_clusters(points, k_clusters)
    last_objective = float('inf')

    # Iterate for a maximum of n_of_iterations
    for _ in range(n_of_iterations):
        # Compute distances from points to centroids
        distances = compute_distances(points, centroid_points)
        assigned_centroids = assign_centroids(distances)
        objective_function_value = calculate_objective(assigned_centroids, distances)
        
        # Check for convergence
        if abs(last_objective - objective_function_value) < error:
            break
        
        # Update centroids
        centroid_points = calculate_new_centroids(points, assigned_centroids, k_clusters)
        last_objective = objective_function_value

    return centroid_points, last_objective

#####################################
#              TASK 01              #
#####################################

# Load data
loaded_points = np.load('data/k_mean_points.npy')

plt.figure()
plt.scatter(loaded_points[:, 0], loaded_points[:, 1])
plt.title("Initial Data Points")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# Specify number of classes
clust_num = 3

# Run K-means clustering
centroid_points, _ = fit(loaded_points, clust_num, n_of_iterations=100)

# Compute final distances and cluster assignments
final_distances = compute_distances(loaded_points, centroid_points)
assigned_clusters = assign_centroids(final_distances)

# Plot clustered data with assigned colors
plt.figure()
for i in range(clust_num):
    cluster_points = loaded_points[assigned_clusters == i]  # Extract points belonging to cluster i
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')

# Plot centroids in black
plt.scatter(centroid_points[:, 0], centroid_points[:, 1], c='black', marker='x', label='Centroids')

# Formatting the plot
plt.title("K-means Clustering Results")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()

#####################################
#              TASK 02              #
#####################################

k_all = range(2, 10)
all_objective = []

# Iteration
for i in k_all:
    _, objective_value = fit(loaded_points, i, n_of_iterations=100)
    all_objective.append(objective_value)

# Plot the objective function values
plt.figure()
plt.plot(k_all, all_objective)
plt.xlabel('K clusters')
plt.ylabel('Sum of squared distance')
plt.show()

#####################################
#              TASK 03              #
#####################################

def compress_image(image: np.ndarray, number_of_colours: int) -> np.ndarray:
    # Compresses the given image by reducing the number of colours used in the image.

    # Get original shape of the image
    original_shape = image.shape
    
    # Reshape image into a 2D array (pixels x color channels)
    reshaped_image = image.reshape((-1, original_shape[2]))

    # Apply K-means clustering to find dominant colours
    kmeans = KMeans(n_clusters=number_of_colours, random_state=42, n_init=10)
    kmeans.fit(reshaped_image)

    # Get cluster centroids and convert them to integers (new colours)
    new_colours = np.round(kmeans.cluster_centers_).astype(np.uint8)

    # Replace original pixel values with their nearest centroid
    compressed_image = new_colours[kmeans.labels_]

    # Reshape back to the original image shape
    compressed_image = compressed_image.reshape(original_shape)

    return compressed_image

loaded_image = plt.imread('data/fish.jpg')

plt.figure()
plt.imshow(loaded_image)
plt.show()

img = compress_image(loaded_image, 30)

plt.figure()
plt.imshow(img)
plt.show()
#%%