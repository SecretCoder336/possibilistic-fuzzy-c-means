import numpy as np
import pandas as pd
import os



np.random.seed(42)

def calculate_centers(data, U, m):
    np.random.seed(42)
    um = U ** m
    centers = np.dot(um.T, data) / np.sum(um, axis=0, keepdims=True).T
    return centers


def update_membership(data, centers, m):
    np.random.seed(42)
    distance_matrix = np.linalg.norm(data[:, np.newaxis] - centers, axis=2) ** (2/(m-1))
    distance_matrix = np.fmax(distance_matrix, np.finfo(np.float64).eps)
    um = 1.0 / distance_matrix
    um = np.fmax(um, np.finfo(np.float64).eps)
    u_new = um / np.sum(um, axis=1, keepdims=True)
    u_new = np.fmax(u_new, np.finfo(np.float64).eps)
    return u_new

def fuzzy_c_means(data, n_clusters, m, max_iters, tol):
    np.random.seed(42)
    n_samples, n_features = data.shape

    
    U = np.random.rand(n_samples, n_clusters)
    U /= np.sum(U, axis=1, keepdims=True)
    
    for iteration in range(max_iters):
        
        U_old = U.copy()

        centers = calculate_centers(data, U, m)
        U = update_membership(data, centers, m)
        diff = np.linalg.norm(U - U_old)
        
        if diff < tol:
            break
    return U, centers


