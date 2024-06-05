import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
import matplotlib.pyplot as plt

class FDModel:
    def __init__(self):
        # Initialize the StandardScaler
        self.scaler = StandardScaler()
        self.mean_vector = None
        self.covariance_matrix = None
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
        data_scaled = self.scaler.fit_transform(data_train)
        
        # Compute the mean vector and covariance matrix of the standardized training data
        self.mean_vector = np.mean(data_scaled, axis=0)
        self.covariance_matrix = np.cov(data_scaled, rowvar=False)
        
        # Add a small regularization term to the diagonal of the covariance matrix to ensure it's invertible
        regularization_term = 1e-5
        self.covariance_matrix += regularization_term * np.eye(self.covariance_matrix.shape[0])
        
        # Compute the Mahalanobis distance for each sample
        mahalanobis_distances = []
        for sample in data_scaled:
            delta = sample - self.mean_vector
            mahalanobis_distance = np.sqrt(np.dot(np.dot(delta.T, np.linalg.inv(self.covariance_matrix)), delta))
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

    def predict(self, data_test):
        """
        Predict the labels for the testing data using the trained model.

        Parameters:
            data_test (numpy.ndarray): Testing data matrix (samples x features).

        Returns:
            numpy.ndarray: Predicted labels for the testing data (0 for normal, 1 for fault).
        """
        # Standardize the testing data using the previously fitted scaler
        data_test_scaled = self.scaler.transform(data_test)
        
        # Compute the Mahalanobis distance for each sample
        mahalanobis_distances = []
        for sample in data_test_scaled:
            delta = sample - self.mean_vector
            mahalanobis_distance = np.sqrt(np.dot(np.dot(delta.T, np.linalg.inv(self.covariance_matrix)), delta))
            mahalanobis_distances.append(mahalanobis_distance)
        
        # Generate prediction labels based on the control limit
        predict_label = np.zeros(len(data_test_scaled))
        for i, distance in enumerate(mahalanobis_distances):
            if distance > self.control_limit:
                predict_label[i] = 1
        
        # Plot the Mahalanobis distances and the control limit
        plt.figure()
        plt.plot(mahalanobis_distances, label='Mahalanobis Distances')
        plt.axhline(y=self.control_limit, color='r', linestyle='--', label='Control Limit')
        plt.title('Mahalanobis Distances for Testing Data')
        plt.xlabel('Sample')
        plt.ylabel('Mahalanobis Distance')
        plt.legend()
        plt.show()
        
        return predict_label
