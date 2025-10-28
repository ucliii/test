import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, find_peaks, detrend
import matplotlib.gridspec as gridspec
import warnings

# 设置中文字体支持，解决警告问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
warnings.filterwarnings("ignore")  # 忽略警告信息

# 1. 读取CSV文件
file_path = r'D:\python_project\data\cleaned_data_2025-10-01_11-07-26.csv'
df = pd.read_csv(file_path, header=None, encoding='utf-8')  

# 2. 提取前12列数据
num_columns = 12  # 处理前12列
num_row = 2500
data_columns = df.iloc[:, :num_columns].values.T  # 转置，使每行代表一列数据
print(f"数据形状: {data_columns.shape} (列数×样本数)")

# 3. 设置采样频率
sampling_rate = 100  # Hz（假设值，请确认你的实际采样率）
print(f"使用的采样率: {sampling_rate} Hz")

# 4. 数据分段函数
def split_channel_data(channel_data):
    """将单个通道数据分成三段"""
    x = len(channel_data)
    
    # 计算分段索引
    segment1 = channel_data[0:x//3-180]
    segment2 = channel_data[x//3+90:2*x//3-90]
    segment3 = channel_data[2*x//3+180:x]
    
    return [segment1, segment2, segment3]

# 5. 设计巴特沃斯带通滤波器 (0.6-4Hz)
def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    设计巴特沃斯带通滤波器
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyq  # 归一化低频截止频率
    high = highcut / nyq  # 归一化高频截止频率
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    应用巴特沃斯带通滤波器
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# 设置滤波器参数
lowcut = 0.5  # 低频截止频率 (Hz)
highcut = 5  # 高频截止频率 (Hz)
order = 5  # 滤波器阶数

# 6. 数据预处理和滤波处理单个分段
def process_single_segment(data_segment, fs, lowcut, highcut, order):
    """处理单个数据分段"""
    # 数据预处理
    if np.any(pd.isnull(data_segment)):
        data_segment = pd.Series(data_segment).fillna(method='ffill').values
    
    # 去除直流分量和线性趋势
    data_detrended = detrend(data_segment - np.mean(data_segment))
    
    # 应用带通滤波器
    filtered_data = butter_bandpass_filter(data_detrended, lowcut, highcut, fs, order)
    
    return data_detrended, filtered_data

# 7. 计算频谱函数
def compute_spectrum(signal, fs):
    """
    计算信号的频谱
    """
    window = np.hanning(len(signal))
    windowed_signal = signal * window
    window_sum = np.sum(window)
    
    fft_result = np.fft.fft(windowed_signal)
    freqs = np.fft.fftfreq(len(fft_result), d=1/fs)
    
    magnitude = np.abs(fft_result) / (window_sum / 2)
    magnitude[0] = magnitude[0] / 2  # 直流分量校正
    
    return freqs, magnitude

# 8. 选择要绘图的列和分段（这里选择第1列第1分段，索引0）
plot_column_index = 8-1  # 可以选择0-11之间的任意列
plot_segment_index = 0   # 选择第1个分段（0, 1, 2）

# 获取要绘图的列数据并分段
original_column_data = data_columns[plot_column_index]
segments = split_channel_data(original_column_data)

# 处理选定的分段
original_seg, filtered_seg = process_single_segment(segments[plot_segment_index], sampling_rate, lowcut, highcut, order)
print(f"第{plot_column_index+1}列 - 分段{plot_segment_index+1}处理完成")

# 9. 绘制4x1子图 - 单个分段的详细对比
fig = plt.figure(figsize=(10, 8))

# 计算时间轴和频谱
time_seg = np.arange(len(original_seg)) / sampling_rate
freqs_orig, mag_orig = compute_spectrum(original_seg, sampling_rate)
freqs_filt, mag_filt = compute_spectrum(filtered_seg, sampling_rate)

# 只关注有意义的频率范围 (0-10Hz)
valid_range = (freqs_orig >= 0) & (freqs_orig <= 10)
freqs_valid = freqs_orig[valid_range]
mag_orig_valid = mag_orig[valid_range]
mag_filt_valid = mag_filt[valid_range]

# 子图1: 原始信号时域波形
ax1 = plt.subplot(4, 1, 1)
ax1.plot(time_seg, original_seg, 'b-', linewidth=1)
ax1.set_title(f'第{plot_column_index+1}列 - 分段{plot_segment_index+1} 原始信号时域波形')
ax1.set_xlabel('时间 (秒)')
ax1.set_ylabel('幅度')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, time_seg[-1])

# 子图2: 滤波后信号时域波形
ax2 = plt.subplot(4, 1, 2)
ax2.plot(time_seg, filtered_seg, 'r-', linewidth=1)
ax2.set_title(f'滤波后信号时域波形 (带通: {lowcut}-{highcut}Hz)')
ax2.set_xlabel('时间 (秒)')
ax2.set_ylabel('幅度')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, time_seg[-1])

# 子图3: 原始信号频域分析
ax3 = plt.subplot(4, 1, 3)
ax3.plot(freqs_valid, mag_orig_valid, 'b-', linewidth=1, label='原始频谱')
ax3.axvspan(lowcut, highcut, color='green', alpha=0.2, label='滤波器通带')
ax3.axvline(lowcut, color='k', linestyle='--', alpha=0.7)
ax3.axvline(highcut, color='k', linestyle='--', alpha=0.7)
ax3.set_title('原始信号频域分析')
ax3.set_xlabel('频率 (Hz)')
ax3.set_ylabel('幅度')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 10)

# 子图4: 滤波后信号频域分析
ax4 = plt.subplot(4, 1, 4)
ax4.plot(freqs_valid, mag_filt_valid, 'r-', linewidth=1, label='滤波后频谱')
ax4.axvspan(lowcut, highcut, color='green', alpha=0.2, label='滤波器通带')
ax4.axvline(lowcut, color='k', linestyle='--', alpha=0.7)
ax4.axvline(highcut, color='k', linestyle='--', alpha=0.7)
ax4.set_title('滤波后信号频域分析')
ax4.set_xlabel('频率 (Hz)')
ax4.set_ylabel('幅度')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 10)

plt.tight_layout()
plt.show()

# 10. 分析该分段的滤波效果
print(f"\n第{plot_column_index+1}列分段{plot_segment_index+1}滤波效果分析:")

# 计算原始信号的频率峰值
peak_indices_orig, _ = find_peaks(mag_orig_valid, height=0.1*np.max(mag_orig_valid), distance=10)
if len(peak_indices_orig) > 0:
    main_freqs_orig = freqs_valid[peak_indices_orig]
    print("  原始信号的主要频率成分:")
    for j, freq in enumerate(main_freqs_orig):
        if lowcut <= freq <= highcut:
            print(f"    峰值 {j+1}: {freq:.2f} Hz (在通带内)")
        else:
            print(f"    峰值 {j+1}: {freq:.2f} Hz (在通带外)")

# 计算滤波后信号的频率峰值
peak_indices_filt, _ = find_peaks(mag_filt_valid, height=0.1*np.max(mag_filt_valid), distance=10)
if len(peak_indices_filt) > 0:
    main_freqs_filt = freqs_valid[peak_indices_filt]
    print("  滤波后信号的主要频率成分:")
    for j, freq in enumerate(main_freqs_filt):
        if lowcut <= freq <= highcut:
            print(f"    峰值 {j+1}: {freq:.2f} Hz (在通带内)")
        else:
            print(f"    峰值 {j+1}: {freq:.2f} Hz (在通带外)")

# 11. 保存所有列的滤波后数据（可选）- 修改为保存分段数据
def save_all_processed_segments(processed_segments_data, sampling_rate, save_path):
    """保存所有处理后的分段数据"""
    # 创建DataFrame
    save_dict = {}
    
    for col_idx, segments in enumerate(processed_segments_data):
        for seg_idx, (original, filtered) in enumerate(segments):
            time_seg = np.arange(len(original)) / sampling_rate
            if col_idx == 0 and seg_idx == 0:  # 只在第一列第一分段保存时间列
                save_dict['Time'] = time_seg
            
            save_dict[f'Col{col_idx+1}_Seg{seg_idx+1}_Original'] = original
            save_dict[f'Col{col_idx+1}_Seg{seg_idx+1}_Filtered'] = filtered
    
    result_df = pd.DataFrame(save_dict)
    result_df.to_csv(save_path, index=False)
    print(f"所有分段数据已保存到: {save_path}")

# 处理所有列的分段数据（用于保存）
all_processed_segments = []
for i in range(num_columns):
    original_column_data = data_columns[i]
    segments = split_channel_data(original_column_data)
    column_segments = []
    for segment in segments:
        original_seg, filtered_seg = process_single_segment(segment, sampling_rate, lowcut, highcut, order)
        column_segments.append((original_seg, filtered_seg))
    all_processed_segments.append(column_segments)
    print(f"第{i+1}列所有分段处理完成")

# 取消注释以下行来保存所有分段数据
# save_all_processed_segments(all_processed_segments, sampling_rate, 
#                           r'D:\python_project\data\all_segments_filtered_data.csv')

print(f"\n处理完成！共处理了{num_columns}列数据，每列分为3段，展示了第{plot_column_index+1}列第{plot_segment_index+1}分段的4x1子图对比。")