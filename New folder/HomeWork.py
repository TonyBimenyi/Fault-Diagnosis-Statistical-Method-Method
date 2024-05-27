import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2

class FDModel:
    
    # 作业完成点1：这里需要定义你的诊断方法相关的变量
    scaler=StandardScaler()
    pca = [] 
    lambda_matrix_invert =[]
    best_PC_Numbers = 20
    control_limit = []

    def __init__(self) -> None:
        pass

    # 作业完成点2：这里需要重写train方法
    def train(self,data_train):        
        data_scaled = self.scaler.fit_transform(data_train)
        n,m=data_scaled.shape
        self.pca = PCA(n_components=self.best_PC_Numbers) 
        data_pca = self.pca.fit_transform(data_scaled)
        lambdas = self.pca.explained_variance_
        lambda_matrix = np.diag(lambdas)
        self.lambda_matrix_invert = np.linalg.inv(lambda_matrix) 

        # 计算控制线
        df = self.best_PC_Numbers  # 例如，自由度是3  
        p = 0.99  # 例如，我们想要找到99%的分位数  
        # 计算卡方分布下使得CDF等于p的x值即为控制线  
        self.control_limit = chi2.ppf(p, df) 

        # 计算T2统计量       
        T2 = []
        for i in range(n):
            T2.append(data_pca[i, :] @ self.lambda_matrix_invert @ data_pca[i, :].T)
        
        # 根据T2和控制线计算检测标签
        predict_label = np.zeros(n)
        for j in range(n):
            if T2[j]>self.control_limit:
                predict_label[j]=1

        return predict_label
    
    # 作业完成点3：这里需要重写predict方法
   