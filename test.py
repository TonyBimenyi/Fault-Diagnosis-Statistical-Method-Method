import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
import matplotlib.pyplot as plt

class FDModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.lambda_matrix_invert = None
        self.best_PC_Numbers = 30
        self.control_limit = None

    def train(self, data_train):

        #Standardize the training data

        data_scaled = self.scaler.fit_transform(data_train)
        n, m = data_scaled.shape

        #Apply PCA to the scaled traing data, retaining the specified number of components
        self.pca = PCA(n_components=self.best_PC_Numbers)
        data_pca = self.pca.fit_transform(data_scaled)

        #compute the variance (lamda) matrix and itsinverse
        lamda = self.pca.explained_variance_
        lamda_matrix = np.diag(lamda)
        self.lambda_matrix_invert = np.linalg.inv(lamda_matrix)

        #compute the control limit using chi-square distribution
        df = self.best_PC_Numbers #degrees of freedon
        p = 0.99 #99%quantile
        self.control_limit = chi2.ppf(p, df)

        #compute the T2 static
        T2 = [data_pca[i, :] @self.lambda_matrix_invert @data_pca[i, :].T for i in range(n)]

        #Generate prediction labels based on the controll llimit
        predict_label =np.zeros(n)
        for j in range(n):
            if T2[j] > self.control_limit:
                predict_label[j] = 1

        
        # Plot the T2 statistics and the control limit
        plt.figure()
        plt.plot(T2, label='T2 Statistics')
        plt.axhline(y=self.control_limit, color='r', linestyle='--', label='Control Limit')
        plt.title('T2 Statistics for Training Data')
        plt.xlabel('Sample')
        plt.ylabel('T2')
        plt.legend()
        plt.show()
        
        return predict_label
