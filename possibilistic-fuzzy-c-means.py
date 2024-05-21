import numpy as np

def initialize_membership_matrix(n_samples, n_clusters):
    membership_matrix = np.random.dirichlet(np.ones(n_clusters), size=n_samples)
    return membership_matrix

def calculate_cluster_centers(data, membership_matrix, m):
    num = np.dot((membership_matrix ** m).T, data)
    den = np.sum(membership_matrix ** m, axis=0)[:, np.newaxis]
    centers = num / den
    return centers

def update_membership_matrix(data, centers, membership_matrix, eta, m, n_clusters):
    dist = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
    dist = np.fmax(dist, np.finfo(np.float64).eps)

    for i in range(n_samples):
        for j in range(n_clusters):
            denom = np.sum([(dist[i, j] / dist[i, k]) ** (2 / (m - 1)) for k in range(n_clusters)])
            membership_matrix[i, j] = 1 / denom

    for i in range(n_samples):
        for j in range(n_clusters):
            t_ij = eta[j] / (eta[j] + dist[i, j])
            membership_matrix[i, j] = (membership_matrix[i, j] ** m) * t_ij

    return membership_matrix

def calculate_eta(data, centers, membership_matrix, m):
    n_clusters = centers.shape[0]
    dist = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
    dist = np.fmax(dist, np.finfo(np.float64).eps)

    eta = np.zeros(n_clusters)
    for j in range(n_clusters):
        eta[j] = np.sum((membership_matrix[:, j] ** m) * dist[:, j]) / np.sum(membership_matrix[:, j] ** m)
    return eta

def pfc_means(data, n_clusters, m=2.0, max_iter=100, tol=1e-4):
    n_samples = data.shape[0]
    membership_matrix = initialize_membership_matrix(n_samples, n_clusters)
    centers = calculate_cluster_centers(data, membership_matrix, m)
    eta = calculate_eta(data, centers, membership_matrix, m)

    for iteration in range(max_iter):
        centers_old = centers.copy()
        membership_matrix = update_membership_matrix(data, centers, membership_matrix, eta, m, n_clusters)
        centers = calculate_cluster_centers(data, membership_matrix, m)
        eta = calculate_eta(data, centers, membership_matrix, m)

        if np.linalg.norm(centers - centers_old) < tol:
            break

    return centers, membership_matrix

# Example usage
data = np.random.rand(100, 2)  # Example data with 100 samples and 2 features
n_clusters = 3
centers, membership_matrix = pfc_means(data, n_clusters)

print("Cluster Centers:\n", centers)
print("Membership Matrix:\n", membership_matrix)
