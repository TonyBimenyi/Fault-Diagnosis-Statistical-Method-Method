#The code utilizes the Mahalanobis distance as a statistical method for fault detection.
#The FDModel class is designed to train and predict using this method.
#The train method trains the model using training data and computes the Mahalanobis distance for each sample.
#The predict method predicts labels for testing data based on the trained model and the computed control limit.
#The added plot_results method visualizes the Mahalanobis distances and control limits for both training and testing data.
#A control limit is computed based on the Chi-square distribution.
#To plot results you need to remove comment from usage example at the bottom of this file

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
        self.train_mahalanobis_distances = []
        self.test_mahalanobis_distances = []

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
        self.train_mahalanobis_distances = []
        for sample in data_scaled:
            delta = sample - self.mean_vector
            mahalanobis_distance = np.sqrt(np.dot(np.dot(delta.T, np.linalg.inv(self.covariance_matrix)), delta))
            self.train_mahalanobis_distances.append(mahalanobis_distance)
        
        # Compute the control limit using chi-square distribution
        df = data_scaled.shape[1]  # degrees of freedom
        p = 0.99  # 99% quantile
        self.control_limit = chi2.ppf(p, df)
        
        # Generate prediction labels based on the control limit
        predict_label = np.zeros(len(data_scaled))
        for i, distance in enumerate(self.train_mahalanobis_distances):
            if distance > self.control_limit:
                predict_label[i] = 1
        
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
        
        # Store the test Mahalanobis distances for plotting
        self.test_mahalanobis_distances.append(mahalanobis_distances)

        # Generate prediction labels based on the control limit
        predict_label = np.zeros(len(data_test_scaled))
        for i, distance in enumerate(mahalanobis_distances):
            if distance > self.control_limit:
                predict_label[i] = 1
        
        return predict_label

    def plot_results(self):
        """
        Plot the Mahalanobis distances and control limits for training and testing data.
        """
        num_plots = 1 + len(self.test_mahalanobis_distances)
        fig, axes = plt.subplots(1, num_plots, figsize=(20, 5))

        # Plot training data Mahalanobis distances
        axes[0].plot(self.train_mahalanobis_distances, label='Mahalanobis Distances')
        axes[0].axhline(y=self.control_limit, color='r', linestyle='--', label='Control Limit')
        axes[0].set_title('Training Data')
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Mahalanobis Distance')
        axes[0].legend()

        # Plot testing data Mahalanobis distances
        for i, distances in enumerate(self.test_mahalanobis_distances):
            axes[i + 1].plot(distances, label='Mahalanobis Distances')
            axes[i + 1].axhline(y=self.control_limit, color='r', linestyle='--', label='Control Limit')
            axes[i + 1].set_title(f'Testing Data Fault {i + 1}')
            axes[i + 1].set_xlabel('Sample')
            axes[i + 1].set_ylabel('Mahalanobis Distance')
            axes[i + 1].legend()

        plt.tight_layout()
        plt.show()

# # Usage example
# import scipy.io as sio

# # Load training data
# matlab_variable = sio.loadmat('./TEP_data/normal.mat')
# data_normal = matlab_variable[list(matlab_variable.keys())[3]]

# # Train the model
# MyFDModel = FDModel()
# predict_label_train = MyFDModel.train(data_normal)

# # Test the model with fault datasets
# test_data_set_numbers = 4
# for i in range(test_data_set_numbers):
#     matlab_variable = sio.loadmat(f'./TEP_data/fault{i + 1}.mat')
#     data_fault = matlab_variable[list(matlab_variable.keys())[3]]
#     MyFDModel.predict(data_fault)

# # Plot the results
# MyFDModel.plot_results()
