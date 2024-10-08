

import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score as sklearn_r2_score




from matplotlib.colors import LinearSegmentedColormap



import gc
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from glob import glob


# 导入 Optuna
import optuna

# GPU 设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def torch_r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def plot_loss(loss):
    # 绘制损失曲线
    plt.figure(figsize=(8, 4))
    x = list(range(len(loss)))
    plt.plot(x, loss, label='total loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.ylim(0, max(loss))
    plt.legend()
    plt.grid(True)
    plt.show()

def custom_loss_v3(X, Y, DeltaW, L, lambda_smooth, alpha, beta):
    # 自定义损失函数
    n = X.shape[0]
    # 计算预测值
    adjusted_pred = torch.sum(DeltaW * X, dim=1, keepdim=True)
    # 计算残差
    residuals = Y - adjusted_pred
    # 数据损失
    data_loss = torch.mean(residuals ** 2)
    # 平滑损失
    product = torch.mm(L, DeltaW)
    smoothness_loss = lambda_smooth * (torch.norm(product, p=1, dim=0).sum())
    # L1 正则化
    l1_loss = alpha * (torch.sum(torch.abs(DeltaW)))
    # L2 正则化
    sqrt_n = torch.sqrt(torch.tensor(n).float())
    l2_loss = beta * (sqrt_n * torch.sum(torch.sqrt((torch.sum(DeltaW ** 2, dim=0)))))
    # 总损失
    total_loss = data_loss + smoothness_loss + l1_loss + l2_loss
    return total_loss

def z_score_normalization(d):
    d = (d - d.min()) / (d.max() - d.min())
    return d

def complete_feature_indices(selected_indices, max_gap=4):
    """
    补全特征索引，如果相邻的特征索引差值小于等于 max_gap，则进行补全。
    
    Args:
        selected_indices (list or np.array): 已经选择的特征索引列表
        max_gap (int): 补全的最大索引差值，默认值为2
    
    Returns:
        np.array: 补全后的特征索引列表
    """
    selected_indices = np.sort(selected_indices)  # 确保索引排序
    completed_indices = set(selected_indices)  # 使用集合以便高效添加补全的索引
    
    for i in range(len(selected_indices) - 1):
        current_idx = selected_indices[i]
        next_idx = selected_indices[i + 1]
        
        # 如果相邻索引的差值小于等于 max_gap，则补全中间的索引
        if next_idx - current_idx <= max_gap:
            for j in range(current_idx + 1, next_idx):
                completed_indices.add(j)
    
    return np.array(sorted(completed_indices))
def complete_feature_indices2(selected_indices, max_gap=4, extend_right=1):
    """
    补全特征索引，如果相邻的三个特征索引（第一和第三个数的差值）小于等于 max_gap，则进行补全并向右延伸。
    
    Args:
        selected_indices (list or np.array): 已经选择的特征索引列表
        max_gap (int): 补全的最大索引差值，默认值为9
        extend_right (int): 向右延伸的数目，默认值为3
    
    Returns:
        np.array: 补全后的特征索引列表
    """
    selected_indices = np.sort(selected_indices)  # 确保索引排序
    completed_indices = set(selected_indices)  # 使用集合以便高效添加补全的索引s
    
    for i in range(len(selected_indices) - 2):
        current_idx = selected_indices[i]
        middle_idx = selected_indices[i + 1]
        next_idx = selected_indices[i + 2]
        
        # 如果相邻三个索引的第一个和第三个之间的差值小于等于 max_gap
        if next_idx - current_idx <= max_gap and (middle_idx - current_idx > 1):
            # 补全第一个和第三个数之间的所有索引
            for j in range(current_idx + 1, next_idx):
                completed_indices.add(j)
            
            # 向右延伸 extend_right 个索引
            for j in range(next_idx + 1, next_idx + 1 + extend_right):
                completed_indices.add(j)
    
    return np.array(sorted(completed_indices))
# 设置数据目录
directory = "/geode2/home/u110/zhou19/BigRed200/spatial_results/data/n=7777"

# 查找匹配的文件
bt_files = glob(os.path.join(directory, "BT_*.csv"))

# 提取基本文件名
base_filenames = [os.path.basename(bt_file).replace("BT_", "").replace(".csv", "") for bt_file in bt_files]
base_filenames = sorted(base_filenames)

# 选择特定的数据集
# loop_numbers=[0,1,2,3,4,5,6,7,8]
# for NUMBER2 in loop_numbers:
NUMBER2 = 8888
chosen_base = base_filenames[NUMBER2]

# 构建完整的文件路径
bt_file = os.path.join(directory, f"BT_{chosen_base}.csv")
h_file = os.path.join(directory, f"H_{chosen_base}.csv")
x_file = os.path.join(directory, f"X_{chosen_base}.csv")
y_file = os.path.join(directory, f"y_{chosen_base}.csv")

# 加载数据
L_norm = pd.read_csv(h_file, header=None)
L_norm = L_norm.T
print(f"L_norm shape:\n{L_norm.shape}\n")

data = pd.read_csv(x_file, header=None)
data = data.T
sample_names = data.index.values
gene_names = data.columns.values
n_samples, n_features = data.shape
print(f"Data shape:\n{data.shape}\n")

target = pd.read_csv(y_file, header=None)
print(f"Target shape:\n{target.shape}\n")

# 计算拉普拉斯矩阵 L
D = np.diag(np.sum(L_norm == 1, axis=1))
A = np.where(L_norm == 1, 1, 0)
L = pd.DataFrame(D - A)

# 转换为 PyTorch 张量
data = torch.from_numpy(data.values).type(torch.float32).to(device)
target = torch.from_numpy(target.values).type(torch.float32).to(device)
L_norm = torch.from_numpy(L.values).type(torch.float32).to(device)

print(f"Loaded files: {bt_file}, {h_file}, {x_file}, {y_file}")
print(f"Chosen base filename: {chosen_base}")

# 数据拆分（K 折交叉验证）
n = n_samples
indices = np.arange(n)

# 打散索引
group_size = n // 5
scattered_indices = np.hstack([indices[i::5] for i in range(5)])

kf = KFold(n_splits=5)
splits_data = []
splits_y = []
splits_H = []
train_indices_list = []
test_indices_list = []

for fold, (train, test) in enumerate(kf.split(scattered_indices)):
    test_indices = scattered_indices[test]
    train_indices = np.setdiff1d(scattered_indices, test_indices)
    
    train_index_tensor = torch.tensor(train_indices, dtype=torch.long)
    train_indices_list.append(train_indices)
    test_index_tensor = torch.tensor(test_indices, dtype=torch.long)
    test_indices_list.append(test_indices)    
    
    data_train, data_test = data[train_index_tensor], data[test_index_tensor]
    y_train, y_test = target[train_index_tensor], target[test_index_tensor]
    H_train = L_norm[train_index_tensor][:, train_index_tensor]
    H_test = L_norm[test_index_tensor][:, test_index_tensor]

    splits_data.append((data_train, data_test))
    splits_y.append((y_train, y_test))
    splits_H.append((H_train, H_test))
    
    print(f"Fold {fold+1}")
    print(f"Train indices: {train_index_tensor}")
    print(f"Test indices: {test_index_tensor}")

# 定义 Optuna 的目标函数
def objective(trial):
    # 采样超参数
    lambda_smooth = trial.suggest_loguniform('lambda_smooth', 1e-4, 0.3)
    alpha = trial.suggest_loguniform('alpha', 1e-4, 0.3)
    beta = trial.suggest_loguniform('beta', 1e-4, 0.3)
    
    # 初始化累加器
    R_score_test = torch.zeros(1).to(device)
    MSE_score_test = torch.zeros(1).to(device)
    
    # 交叉验证循环
    for m, (data_split, y_split, H_split) in enumerate(zip(splits_data, splits_y, splits_H)):
        # 训练数据
        data_train = data_split[0].to(device)
        y_train = y_split[0].to(device)
        H_train = H_split[0].to(device)
        
        # 测试数据
        data_test = data_split[1].to(device)
        y_test = y_split[1].to(device)
        H_test = H_split[1].to(device)
        
        n_train, p = data_train.shape

        # 初始化参数
        DeltaW = torch.randn(n_train, p, device=device, requires_grad=True)
        learning_rate = 3e-2
        convergence_threshold = 0.1
        convergence_count = 0
        max_convergence_count = 20
        num_iterations = 20000

        optimizer = torch.optim.SGD([DeltaW], lr=learning_rate)
        all_loss = []
        out_count = 0

        # 优化循环
        while convergence_count < max_convergence_count:
            for i in range(num_iterations):
                optimizer.zero_grad()
                loss = custom_loss_v3(data_train, y_train, DeltaW, H_train, lambda_smooth, alpha, beta)
                loss.backward()
                optimizer.step()
                all_loss.append(loss.item())
                if i > 0:
                    if abs(all_loss[-1] - all_loss[-2]) < convergence_threshold:
                        convergence_count += 1
                    else:
                        convergence_count = 0
                    if convergence_count >= max_convergence_count:
                        break
            else:
                break

            # 如果收敛较慢，调整学习率
            if convergence_count < max_convergence_count * 0.5:
                if len(all_loss) > 200 and abs(np.mean(all_loss[-201:-101]) - np.mean(all_loss[-101:-1])) < 0.1:
                    convergence_count = max_convergence_count
                    break
                else:
                    learning_rate *= 2
                    out_count += 1
                    if out_count > 2:
                        convergence_count = max_convergence_count
                        break

        # 获取优化后的参数
        delta_beta = DeltaW.detach()

        # 计算测试集上的预测
        # 为简化，使用 DeltaW 的平均值作为全局参数
        delta_beta_mean = delta_beta.mean(dim=0, keepdim=True)

        y_hat_test = (data_test * delta_beta_mean).sum(dim=1)
        MSE_test = torch.mean((y_test.squeeze() - y_hat_test) ** 2)
        R_score_t = torch_r2_score(y_test.squeeze(), y_hat_test)

        MSE_score_test += MSE_test
        R_score_test += R_score_t

    # 计算平均得分
    avg_R_score_test = R_score_test.item() / len(splits_data)
    avg_MSE_score_test = MSE_score_test.item() / len(splits_data)
    
    # 返回需要最小化的指标（这里是平均 MSE）
    return avg_MSE_score_test



import time

# 开始计时
start_time = time.time()

# 运行 Optuna 优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # 可以根据需要调整 n_trials
# 结束计时

# 输出最优超参数
print('Best trial:')
trial = study.best_trial

print(f'  MSE: {trial.value}')
print('  Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

# 可视化优化历史（可选）
# from optuna.visualization import plot_optimization_history, plot_param_importances
# plot_optimization_history(study)
# plot_param_importances(study)


print("All is done!!!!!!!!!!!!!!!!!!!!!!!!!!")


params = trial.params

# 单独获取每个参数
lambda_smooth = params['lambda_smooth']
alpha = params['alpha']
beta = params['beta']

################################################################################################################################################################################################
################################################################################################################################################################################################
############################################################
#############################################################                                                                   ####################################################################################
##############################################################                     Start get results                                                  #################################################################
###############################################################                                                                           ###########################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################

# lambda_smooth= 0.04075774781067626
# alpha= 0.00010497753885602024
# beta= 0.008338836356549863



# numbers=[NUMBER2]#range(9,27)
# for mm in numbers:

    # Define the path to the directory containing the files
    # directory = "F:/OneDrive - Indiana University/shared/spatial_python_pd/final_code_data/n=1000"


    
mm=NUMBER2
# Find all files in the folder that match the BT_*.csv pattern
bt_files = glob(os.path.join(directory, "BT_*.csv"))

# Extract the base filenames by removing the BT_ prefix and .csv suffix
base_filenames = [os.path.basename(bt_file).replace("BT_", "").replace(".csv", "") for bt_file in bt_files]

# Sort the base_filenames to ensure a consistent order
base_filenames = sorted(base_filenames)

# Now, base_filenames contains all the 27 different combinations
# You can access them using base_filenames[index], for example:
# print(base_filenames)  # This will show the list of all base filenames
# Example: To load the first set (base_filenames[0])
############################################
chosen_base = base_filenames[mm]  # You can change the index to access a different set
dir11 =os.path.join(directory, 'results')

directory2 = os.path.join(dir11, chosen_base)

# 创建文件夹
os.makedirs(directory2, exist_ok=True)
#os.chdir(directory2)

# Construct the full filenames based on the chosen base
bt_file = os.path.join(directory, f"BT_{chosen_base}.csv")
h_file = os.path.join(directory, f"H_{chosen_base}.csv")
x_file = os.path.join(directory, f"X_{chosen_base}.csv")
y_file = os.path.join(directory, f"y_{chosen_base}.csv")
# hyperparameter_file = os.path.join(directory, f"combined___{chosen_base}.csv")
# Load the files using pandas
# BT = pd.read_csv(bt_file, header=None)
# hyperparameter = pd.read_csv(hyperparameter_file, header=None)
L_norm = pd.read_csv(h_file, header=None)
L_norm=L_norm.T
print(f"l_norm shape:\n{L_norm.shape}\n")

data = pd.read_csv(x_file, header=None)
data=data.T
sample_names = data.index.values
gene_names = data.columns.values
n_samples,n_features = data.shape
print(f"data shape:\n{data.shape}\n")

target = pd.read_csv(y_file, header=None)
print(f"target shape:\n{target.shape}\n")

D = np.diag(np.sum(L_norm == 1, axis=1))
A = np.where(L_norm == 1, 1, 0)
# Laplacian matrix  L
L = pd.DataFrame(D - A)
data = torch.from_numpy(data.values).type(torch.float32).to(device)
target = torch.from_numpy(target.values).type(torch.float32).to(device)
L_norm = torch.from_numpy(L.values).type(torch.float32).to(device)

base_tag = chosen_base.split('_')
# Optionally, print a confirmation of the loaded files
# print(f"Loaded files: {bt_file}, {h_file}, {x_file}, {y_file}")
print(chosen_base)
# hyperparameter_sorted = hyperparameter.sort_values(by=hyperparameter.columns[0])
ari_manual=0
best = "New CV"

# best_tag = "BEST_"+str(best)
# choose_line = best
# lambda_smooth = float(hyperparameter_sorted.iloc[choose_line, 1])ssss
# alpha = float(hyperparameter_sorted.iloc[choose_line, 2])
# beta = float(hyperparameter_sorted.iloc[choose_line, 3])


# while ari_manual<0.7:

best_tag = "BEST_"+str(best)
choose_line = best
# lambda_smooth = float(hyperparameter_sorted.iloc[choose_line, 1])
# alpha = float(hyperparameter_sorted.iloc[choose_line, 2])
# beta = float(hyperparameter_sorted.iloc[choose_line, 3])
# beta = 0.010813721488071857
# alpha = 0.00010407556884845393
# lambda_smooth = 0.03396043293415434
# if (base_tag[4]=='1')&(best==5):
    
#     chosen_base22 = base_filenames[mm-1]  # You can change the index to access a different set
#     dir1122 =os.path.join(directory, 'results')
    
#     directory222 = os.path.join(dir1122, chosen_base22)
    

#     hyperparameter_file22 = os.path.join(directory, f"combined___{chosen_base}.csv")
#     hyperparameter22 = pd.read_csv(hyperparameter_file, header=None)
#     hyperparameter_sorted22 = hyperparameter.sort_values(by=hyperparameter.columns[0])
#     best_tag = "BEST_"+str(best)
#     choose_line = 0
#     lambda_smooth = float(hyperparameter_sorted22.iloc[choose_line, 1])
#     alpha = float(hyperparameter_sorted22.iloc[choose_line, 2])
#     beta = float(hyperparameter_sorted22.iloc[choose_line, 3])
#     print("use 0.5")
# if (base_tag[4]=='2')&(best==5):
    
#     chosen_base22 = base_filenames[mm-2]  # You can change the index to access a different set
#     dir1122 =os.path.join(directory, 'results')
    
#     directory222 = os.path.join(dir1122, chosen_base22)
    

#     hyperparameter_file22 = os.path.join(directory, f"combined___{chosen_base}.csv")
#     hyperparameter22 = pd.read_csv(hyperparameter_file, header=None)
#     hyperparameter_sorted22 = hyperparameter.sort_values(by=hyperparameter.columns[0])
#     best_tag = "BEST_"+str(best)
#     choose_line = 0
#     lambda_smooth = float(hyperparameter_sorted22.iloc[choose_line, 1])
#     alpha = float(hyperparameter_sorted22.iloc[choose_line, 2])
#     beta = float(hyperparameter_sorted22.iloc[choose_line, 3])
#         # 初始化超参数
#     print("use 0.5")
    

learning_rate = 2e-2
num_iterations = 5000
initial_threshold = 1  # 设置初始阈值较高

if (n_samples<n_features)|(n_features>199):  #    high D  or too much feature
    # 定义交叉验证的拆分策略
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    selected_features_per_fold = []

    # 执行交叉验证
    for train_index, val_index in kf.split(data):
        print(f"Processing fold with train indices: {train_index}, validation indices: {val_index}")
        print(len(train_index))
        # 根据当前折的训练数据进行特征选择
        current_data = data[train_index, :]  #. X
        current_target = target[train_index] #.  y
        n_samples2 = len(train_index)
        current_features_indices = np.arange(n_features)  # 保留原始特征的索引
        threshold = initial_threshold
        L_norm2=L_norm[train_index, :][:, train_index]
        hold=0
        # flag=0
        while (current_data.shape[1] > (0.7*n_samples2))|(current_data.shape[1]>199):
    #         DeltaW = torch.randn(n_samples2, current_data.shape[1], device=device, requires_grad=True)
            DeltaW = torch.randn(current_data.shape[0], current_data.shape[1], device=device, requires_grad=True)
            optimizer = torch.optim.Adam([DeltaW], lr=learning_rate)
            all_loss = []
            for i in range(num_iterations):
                optimizer.zero_grad()
                loss = custom_loss_v3(current_data, current_target, DeltaW, L_norm2, lambda_smooth, alpha, beta)
                loss.backward()
                optimizer.step()
                all_loss.append(loss.item())

            # 计算每个特征的绝对值之和，并找出所有绝对值之和小于当前阈值的特征
            feature_sums = torch.sum(torch.abs(DeltaW), dim=0)
            save_features_indices = torch.where(feature_sums >= threshold)[0].cpu().numpy()
            # 更新当前的特征索引列表和数据矩阵
            current_features_indices = current_features_indices[save_features_indices]
            current_data = current_data[:, save_features_indices]

            delta_beta = DeltaW[:, save_features_indices].cpu().detach().numpy()
            delta_beta_df = pd.DataFrame(delta_beta, index=sample_names[train_index], columns=gene_names[current_features_indices])

            # 动态调整阈值：根据当前特征数量调整阈值增长策略

            if hold > 6:
                # flag=1
                break

            if len(save_features_indices) > 3 * n_samples2:
                threshold += 2
            elif len(save_features_indices) > 1.3 * n_samples2:
                threshold += 1
            else:
                hold=hold+1
                threshold += 0.2

            print(f"Increasing threshold to {threshold}")

        # 保存当前折中保留的特征索引
        selected_features_per_fold.append(current_features_indices)

    # 综合所有折的结果
    # 可以选择在大多数折中都被选到的特征，或者取并集、交集等策略
    from collections import Counter
    all_selected_features = np.concatenate(selected_features_per_fold)
    feature_counts = Counter(all_selected_features)
    final_selected_features = [feature for feature, count in feature_counts.items() if count >= 4]  # 在至少3个折中被选择的特征

    print(f"Final 1 selected features after cross-validation: {final_selected_features}")
    if len(final_selected_features)>0.6*(n_samples):
        current_features_indices=np.sort(final_selected_features)
    else:
        #    补充之后不应该超过sample
        completed_indices = complete_feature_indices2(np.sort(final_selected_features))
        current_features_indices = completed_indices
        print(f"Final 2 selected features after cross-validation: {current_features_indices}")

else:
    current_features_indices = np.arange(n_features)
    
    


        # 初始化 DeltaW
DeltaW2 = torch.randn(n_samples, len(current_features_indices), device=device, requires_grad=True)
data2 = data[:, current_features_indices]

learning_rate = 2e-2
num_iterations = 23000
# 梯度下降优化器
optimizer = torch.optim.Adam([DeltaW2], lr=learning_rate)

all_loss = []
for i in range(num_iterations):
    optimizer.zero_grad()
    loss = custom_loss_v3(data2, target, DeltaW2, L_norm, lambda_smooth, alpha, beta)
    loss.backward()
    optimizer.step()
    all_loss.append(loss.item())
# plt.savefig(os.path.join(directory2, f"{best_tag}_##Loss_{chosen_base}.pdf"),format='pdf', bbox_inches='tight')
# plot_loss(all_loss)
# plt.close()

########
delta_beta2 = DeltaW2.cpu().detach().numpy()
delta_beta2 = pd.DataFrame(delta_beta2,index=sample_names,columns=gene_names[current_features_indices])

colors = ["blue", "white", "red"]  
n_bins =   n_samples
cmap_name = "my_cmap"
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

plt.figure(figsize=(15, 12))
sns.heatmap(delta_beta2.T, cmap=cm, annot=False, center=0)
plt.savefig(os.path.join(directory2, f"{best_tag}_##BT_before_{chosen_base}.pdf"),format='pdf', bbox_inches='tight')
#/Users/xinyuzhou/Library/CloudStorage/OneDrive-IndianaUniversity/shared/spatial_python_pd/final_code_data/n=200
# plt.show()
plt.close()
################
# 计算每个特征的绝对值之和，并找出所有绝对值之和小于1的特征
feature_sums2 = torch.sum(torch.abs(DeltaW2), dim=0)
zero_or_small_features_indices2 = torch.where(feature_sums2 < 0.5)[0].cpu().numpy()
save_features_indices_new = torch.where(feature_sums2 >= 0.5)[0].cpu().numpy()
# 打印并检查这些特征的索引
print(len(zero_or_small_features_indices2))
if (n_samples<n_features)|(n_features>199):  
    # 初始化一个全零矩阵，其大小与原始数据矩阵相同
    original_matrix_shape = (data.shape[1], data.shape[0])  # n_features 是原始数据的特征数
    restored_matrix = np.zeros(original_matrix_shape)

    # 将保留下来的特征行放回到原矩阵中
    restored_matrix[current_features_indices, :] = delta_beta2.T
    plt.figure(figsize=(15, 12))
    sns.heatmap(restored_matrix, cmap=cm, annot=False, center=0)
    plt.savefig(os.path.join(directory2, f"{best_tag}_##BT_restore_{chosen_base}.pdf"),format='pdf', bbox_inches='tight')

    # plt.show()
    plt.close()
else:
    restored_matrix = delta_beta2.T
delta_beta_df_re = pd.DataFrame(restored_matrix)
delta_beta_path = os.path.join(directory2, f"{best_tag}_##BT_restore_{chosen_base}.csv")
delta_beta_df_re.to_csv(delta_beta_path)

######### true
true_B= pd.read_csv(bt_file, header=None)
# true_B=true_B.iloc[1:,]
DeltaW_true = torch.from_numpy(true_B.values).type(torch.float32).to(device)
colors = ["blue", "white", "red"]  
n_bins = n_samples  
cmap_name = "my_cmap"
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

plt.figure(figsize=(15, 12))
sns.heatmap(true_B, cmap=cm, annot=False, center=0)
plt.savefig(os.path.join(directory2, f"{best_tag}_##BT_true_{chosen_base}.pdf"),format='pdf', bbox_inches='tight')

# plt.show()
plt.close()




###  get the clusters
matrix1 = true_B

# 准备数据，将矩阵展平
data_flat = matrix1.values.flatten().reshape(-1, 1)

# 确定最佳聚类数（使用肘部法则）
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(data_flat)
    inertia.append(kmeans.inertia_)

# 绘制肘部法则图
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.savefig(os.path.join(directory2, f"{best_tag}_##BElbow_true_{chosen_base}.pdf"),format='pdf', bbox_inches='tight')

# plt.show()
plt.close()

optimal_clusters = int(base_tag[2])+1
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(data_flat)

# 将聚类结果重新映射回矩阵的形状
clustered_matrix1 = clusters.reshape(matrix1.shape)

# 保存结果到新的 CSV 文件
clustered_matrix1_df = pd.DataFrame(clustered_matrix1, index=matrix1.index, columns=matrix1.columns)
output_file_path = os.path.join(directory2, f"{best_tag}_##Cluster_true_{chosen_base}.csv")
clustered_matrix1_df.to_csv(output_file_path)

##  get the presiction clusters
# 读取数据
matrix2 = delta_beta_df_re

# 准备数据，将矩阵展平
data_flat = matrix2.values.flatten().reshape(-1, 1)

# 确定最佳聚类数（使用肘部法则）
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(data_flat)
    inertia.append(kmeans.inertia_)

# 绘制肘部法则图
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.savefig(os.path.join(directory2, f"{best_tag}_##Elbow_prediction_{chosen_base}.pdf"),format='pdf', bbox_inches='tight')

# plt.show()
plt.close()

optimal_clusters = int(base_tag[2])+1
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(data_flat)

# 将聚类结果重新映射回矩阵的形状
clustered_matrix2 = clusters.reshape(matrix2.shape)

# 保存结果到新的 CSV 文件
clustered_matrix2_df = pd.DataFrame(clustered_matrix2, index=matrix2.index, columns=matrix2.columns)
output_file_path2 = os.path.join(directory2, f"{best_tag}_##Cluster_prediction_{chosen_base}.csv")
clustered_matrix2_df.to_csv(output_file_path2)
##   get the ARI

import numpy as np
from scipy.special import comb
matrix1_np = clustered_matrix1_df.to_numpy()
matrix2_np = clustered_matrix2_df.to_numpy()
def adjusted_rand_index(labels_true, labels_pred):
    """Manually calculate the Adjusted Rand Index."""
    contingency = np.histogram2d(labels_true, labels_pred, bins=(len(np.unique(labels_true)), len(np.unique(labels_pred))))[0]
    sum_comb_c = sum(comb(n_c, 2) for n_c in np.sum(contingency, axis=1))
    sum_comb_k = sum(comb(n_k, 2) for n_k in np.sum(contingency, axis=0))
    sum_comb = sum(comb(n_ij, 2) for n_ij in contingency.flatten())
    n = np.sum(contingency)
    comb_n = comb(n, 2)
    
    index = sum_comb
    expected_index = (sum_comb_c * sum_comb_k) / comb_n
    max_index = (sum_comb_c + sum_comb_k) / 2
    
    return (index - expected_index) / (max_index - expected_index)

# 使用手动实现的函数来计算 ARI
labels_true = matrix1_np.flatten()
labels_pred = matrix2_np.flatten()
ari_manual = adjusted_rand_index(labels_true, labels_pred)
print("Manual ARI score:", ari_manual)
end_time = time.time()

# 计算执行时间并转换为分钟
run_time_minutes = (end_time - start_time) / 60


with open(os.path.join(directory2, f"{best_tag}_##Hyperparameter_ARI_{chosen_base}.txt"), 'w') as file:
    # 写入每个变量到文件中
    file.write(f'lambda_smooth = {lambda_smooth}\n')
    file.write(f'alpha = {alpha}\n')
    file.write(f'beta = {beta}\n')
    file.write(f'Times = {run_time_minutes}\n')
    file.write(f'ARI = {ari_manual}\n')
    
    
# if (best>4)& (ari_manual<0.7):
#     ari_manual=100
# best=best+1



    


# 计算残差矩阵
residual_matrix = matrix1_np - matrix2_np

# 将残差矩阵转换为 DataFrame（如果需要）
residual_df = pd.DataFrame(residual_matrix)

# 绘制残差热图
plt.figure(figsize=(15, 12))
sns.heatmap(residual_df, cmap='coolwarm', annot=False, center=0)
# plt.title('Residual Heatmap')
plt.savefig(os.path.join(directory2, f"{best_tag}_##residual_{chosen_base}.pdf"),format='pdf', bbox_inches='tight')

# plt.show()
plt.close()

# 调用垃圾回收
gc.collect()

# 如果使用 GPU，清空 GPU 缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()




