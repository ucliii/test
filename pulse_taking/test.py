import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings

# 设置中文字体支持，解决警告问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
warnings.filterwarnings("ignore")  # 忽略警告信息

# 1. 读取CSV文件
file_path = r'D:\python_project\data\cleaned_data_2025-09-19_10-16-28.csv'
df = pd.read_csv(file_path, header=None, encoding='utf-8')  

# 2. 提取指定列数据
column_index = 11  # 第12列（索引从0开始）
data = df.iloc[:, column_index].values

print(f"数据长度: {len(data)} 个样本点")

# 3. 数据预处理
if np.any(pd.isnull(data)):
    print("数据中存在缺失值，正在填充...")
    data = pd.Series(data).fillna(method='ffill').values

# 3.5 去除直流分量和线性趋势
data_detrended = data - np.mean(data)
# 更彻底的去除趋势
from scipy.signal import detrend
data_detrended = detrend(data_detrended)

# 4. 确定正确的采样频率 - 这是关键！
# 你需要知道实际的时间信息，例如：
# - 总采集时间是多少秒？
# - 或者采样间隔是多少秒？

# 假设你不知道确切采样率，我们可以估算
# 从时域信号估算脉搏率
height_threshold = 0.5*np.std(data_detrended)  # 增加高度阈值
distance_threshold = 50  # 最小峰值间距（样本点）
peak_indices, peak_properties = find_peaks(
    data_detrended, 
    height=height_threshold,  # 增加高度阈值
    distance=distance_threshold,  # 添加最小间距
    prominence=0.5  # 添加显著性要求
)
if len(peak_indices) > 1:
    avg_peak_interval = np.mean(np.diff(peak_indices))
    print(f"平均脉搏间隔: {avg_peak_interval} 样本点")
    
    # 如果你知道总采集时间，可以估算采样率
    # total_time = 60  # 假设采集了60秒
    # sampling_rate = len(data) / total_time
    # print(f"估算采样率: {sampling_rate:.2f} Hz")
else:
    print("无法检测到足够多的脉搏峰值")

# 5. 可视化数据及峰值
plt.figure(figsize=(12, 6))
plt.plot(data_detrended, label='去趋势信号', color='blue', linewidth=1)
plt.plot(peak_indices, data_detrended[peak_indices], ".", 
         label='检测到的峰值', color='red', markersize=8, markeredgewidth=2)

# 添加标注
for i, peak in enumerate(peak_indices):
    plt.text(peak, data_detrended[peak] + 0.1, f'{i+1}', 
             ha='center', va='bottom', fontsize=8, color='darkgreen')

plt.title('去趋势信号与检测到的峰值', fontsize=14)
plt.xlabel('样本点索引', fontsize=12)
plt.ylabel('信号幅度', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()