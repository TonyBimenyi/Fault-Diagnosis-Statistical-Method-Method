# 故障诊断技术结课作业说明
# 请同学们提交一个名为HomeWork.py的文件
# 该文件中定义了一个名为FDModel的类
# 该类中包含了名为train和predict的方法
# train方法仅输入训练数据，返回训练数据的预测标签（0代表正常，1代表故障）
# predict方法仅输入测试数据，返回测试数据的预测标签（0代表正常，1代表故障）
# 然后执行TestHomeWork，可得到 Average of Acc
# 最终成绩根据Average of Acc排名

# 请勿修改TestHomeWork中的任何内容，训练和测试过程及结果完全封装在FDModel中
# 执行本演示示例需配置numpy、matplotlib、sklearn工具包，scipy一般会自动关联安装

# 本文档所在文件夹下包含了一个名为TEP_data的文件夹，里面是训练和测试数据，包含1个正常数据集和fault1-fault4
# 上述5个文件用于模型开发和代码测试
# 最终成绩将由总计28个故障的测试准确率综合计算

# Dear students,
# Please submit a file named 'HomeWork.py'.
# In this file, you are required to define a class named 'FDModel'.
# This class should contain two methods: 'train' and 'predict'.
# The 'train' method should take only training data as input and return the predicted labels for the training data (0 represents normal, 1 represents fault).
# The 'predict' method should take only testing data as input and return the predicted labels for the testing data (0 represents normal, 1 represents fault).
# After that, by executing 'TestHomeWork', you can obtain the 'Average of Acc'.
# Your final score will be ranked based on the 'Average of Acc'.
# Please do not modify any content in 'TestHomeWork'. The training, testing processes, and results should be fully encapsulated in 'FDModel'.
# To run this demonstration, you need to have numpy, matplotlib, and sklearn toolkits installed. scipy is generally installed automatically as a dependency.
# Under the folder where this document is located, there is a folder named 'TEP_data' containing training and testing data, which include one normal dataset and fault1 to fault4.
# These 5 files will be used for model development and code testing.
# The final score will be calculated based on the average testing accuracy of a total of 28 fault scenarios.

import scipy.io as sio
import numpy as np
from HomeWork import FDModel
import matplotlib.pyplot as plt

# 1、准备数据
matlab_variable = sio.loadmat('./TEP_data/normal.mat')
matlab_variable_keys = list(matlab_variable.keys())
matrix_name = matlab_variable_keys[3]# .mat文件中第3个（0起始）key为数据矩阵名,据此名称将数据导出为numpy矩阵
data_normal = matlab_variable[matrix_name] 
n,m=data_normal.shape
print(matrix_name+" contains totally "+str(n)+" samples "+"and "+str(m)+" variables for training")

# 2、训练
MyFDModel = FDModel()
PC_numbers = 30
predict_label_train = MyFDModel.train(data_normal)

# 3、测试
matrix_name = []
predict_label_test = {}
test_data_set_numbers = 4
for i in range(test_data_set_numbers):
    matlab_variable = sio.loadmat('./TEP_data/fault'+str(i+1)+'.mat')
    matlab_variable_keys = list(matlab_variable.keys())
    matrix_name.append(matlab_variable_keys[3]) # .mat文件中第3个（0起始）key为数据矩阵名,据此名称将数据导出为numpy矩阵
    data_fault = matlab_variable[matrix_name[i]] 
    n,m=data_fault.shape
    print(matrix_name[i]+" contains totally "+str(n)+" samples "+"and "+str(m)+" variables for testing")
    predict_label_test[matrix_name[i]] = MyFDModel.predict(data_fault)
    plt.subplot(1, test_data_set_numbers, i+1)
    plt.plot(predict_label_train)
    plt.plot(predict_label_test[matrix_name[i]])
plt.show()
# 4、评估
# 故障数据集中前160个是正常的，后800个是故障
true_label = np.concatenate((np.zeros(160),np.ones(800)))
accs = []
for i in range(test_data_set_numbers):
    y_true = np.array(true_label)  
    y_pred = np.array(predict_label_test[matrix_name[i]])        
    # 检查形状是否匹配  
    assert y_true.shape == y_pred.shape, "y_true和y_pred的形状必须相同"        
    # 计算正确分类的样本数  
    correct_predictions = np.sum(y_true == y_pred)        
    # 计算总样本数  
    total_samples = len(y_true)        
    # 计算准确率  
    accuracy = correct_predictions / total_samples 
    accs.append(accuracy)
    print("Acc of fault "+str(i+1)+" is "+str(accuracy))
print("Average of Acc is "+str(np.mean(accs)))
