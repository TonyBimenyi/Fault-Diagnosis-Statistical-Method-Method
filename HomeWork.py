import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

class FDModel:
    def __init__(self):
        # Initialize the StandardScaler
        self.scaler = None
        self.control_limit = None

    def train(self, data_train):
        """
        Train the fault detection model using statistical methods.

        Parameters:
            data_train (numpy.ndarray): Training data matrix (samples x features).

        Returns:
            numpy.ndarray: Predicted labels for the training data (0 for normal, 1 for fault).
        """
        # Standardize the training data
        self.scaler = StandardScaler()
        data_scaled = self.scaler.fit_transform(data_train)
        
        # Compute the mean vector and covariance matrix of the standardized training data
        mean_vector = np.mean(data_scaled, axis=0)
        covariance_matrix = np.cov(data_scaled.T)
        
        # Compute the Mahalanobis distance for each sample
        mahalanobis_distances = []
        for sample in data_scaled:
            delta = sample - mean_vector
            mahalanobis_distance = np.sqrt(np.dot(np.dot(delta.T, np.linalg.inv(covariance_matrix)), delta))
            mahalanobis_distances.append(mahalanobis_distance)
        
        # Compute the control limit using chi-square distribution
        df = data_scaled.shape[1]  # degrees of freedom
        p = 0.99  # 99% quantile
        self.control_limit = chi2.ppf(p, df)
        
        # Generate prediction labels based on the control limit
        predict_label = np.zeros(len(data_scaled))
        for i, distance in enumerate(mahalanobis_distances):
            if distance > self.control_limit:
                predict_label[i] = 1
            else:
                predict_label[i] = 0

        
        # Plot the Mahalanobis distances and the control limit
        plt.figure()
        plt.plot(mahalanobis_distances, label='Mahalanobis Distances')
        plt.axhline(y=self.control_limit, color='r', linestyle='--', label='Control Limit')
        plt.title('Mahalanobis Distances for Training Data')
        plt.xlabel('Sample')
        plt.ylabel('Mahalanobis Distance')
        plt.legend()
        plt.show()
        
        return predict_label
