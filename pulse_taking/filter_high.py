import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, find_peaks
import matplotlib.gridspec as gridspec
import warnings

# 设置中文字体支持，解决警告问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
warnings.filterwarnings("ignore")  # 忽略警告信息

# 1. 读取CSV文件
file_path = r'D:\python_project\data\cleaned_data_2025-10-01_11-07-26.csv'
df = pd.read_csv(file_path, header=None, encoding='utf-8')  

# 2. 提取指定列数据
column_index = 8  # 第12列（索引从0开始）
data = df.iloc[:, column_index].values

print(f"数据长度: {len(data)} 个样本点")

# 3. 数据预处理
if np.any(pd.isnull(data)):
    print("数据中存在缺失值，正在填充...")
    data = pd.Series(data).fillna(method='ffill').values

# 去除直流分量和线性趋势
from scipy.signal import detrend
data_detrended = detrend(data - np.mean(data))

# 4. 设置采样频率（需要根据你的实际情况调整！）
sampling_rate = 100  # Hz（假设值，请确认你的实际采样率）
print(f"使用的采样率: {sampling_rate} Hz")

# 5. 设计巴特沃斯带通滤波器 (0.6-4Hz)
def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    设计巴特沃斯带通滤波器
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyq  # 归一化低频截止频率
    high = highcut / nyq  # 归一化高频截止频率
    b, a = butter(order, [low, high], btype='band')  # 设计带通滤波器[1,3,5](@ref)
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    应用巴特沃斯带通滤波器
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # 使用filtfilt进行零相位滤波[3,4](@ref)
    y = filtfilt(b, a, data)
    return y

# 设置滤波器参数
lowcut = 0.5  # 低频截止频率 (Hz)
highcut = 4.0  # 高频截止频率 (Hz)
order = 5  # 滤波器阶数[1,3](@ref)

# 应用带通滤波器
filtered_data = butter_bandpass_filter(data_detrended, lowcut, highcut, sampling_rate, order)

# 6. 计算滤波前后信号的频谱
def compute_spectrum(signal, fs):
    """
    计算信号的频谱
    """
    # 应用窗函数减少频谱泄漏
    window = np.hanning(len(signal))
    windowed_signal = signal * window
    window_sum = np.sum(window)
    
    # 计算FFT
    fft_result = np.fft.fft(windowed_signal)
    freqs = np.fft.fftfreq(len(fft_result), d=1/fs)
    
    # 计算幅度谱
    magnitude = np.abs(fft_result) / (window_sum / 2)
    magnitude[0] = magnitude[0] / 2  # 直流分量校正
    
    return freqs, magnitude

# 计算原始信号和滤波后信号的频谱
freqs_orig, mag_orig = compute_spectrum(data_detrended, sampling_rate)
freqs_filt, mag_filt = compute_spectrum(filtered_data, sampling_rate)

# 只关注有意义的频率范围 (0-10Hz)
valid_range = (freqs_orig >= 0) & (freqs_orig <= 10)
freqs_orig = freqs_orig[valid_range]
mag_orig = mag_orig[valid_range]
mag_filt = mag_filt[valid_range]

# 7. 可视化结果
plt.figure(figsize=(15, 8))
gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])

# 7.1 绘制滤波器频率响应
b, a = butter_bandpass(lowcut, highcut, sampling_rate, order)
w, h = freqz(b, a, worN=2000, fs=sampling_rate)

plt.subplot(gs[0, :])
plt.plot(w, np.abs(h), 'b', linewidth=2)
plt.axvline(lowcut, color='k', linestyle='--', alpha=0.7)
plt.axvline(highcut, color='k', linestyle='--', alpha=0.7)
plt.title('巴特沃斯带通滤波器频率响应 (0.6-4Hz)')
plt.xlabel('频率 (Hz)')
plt.ylabel('增益')
plt.grid(True)
plt.xlim(0, 10)

# 7.2 绘制时域信号对比
# 原始信号
ax1 = plt.subplot(gs[1, 0])
time = np.arange(len(data_detrended)) / sampling_rate
plt.plot(time, data_detrended, 'b-', label='原始信号', alpha=0.7)
plt.title('滤波前信号 (时域)')
plt.xlabel('时间 (秒)')
plt.ylabel('幅度')
plt.legend()
plt.grid(True)

# 滤波后信号
ax2 = plt.subplot(gs[1, 1])
plt.plot(time, filtered_data, 'r-', label='滤波后信号', alpha=0.7)
plt.title('滤波后信号 (时域)')
plt.xlabel('时间 (秒)')
plt.ylabel('幅度')
plt.legend()
plt.grid(True)

# 设置相同的y轴范围以便比较
y_min = min(np.min(data_detrended), np.min(filtered_data))
y_max = max(np.max(data_detrended), np.max(filtered_data))
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)

# 7.3 绘制频域信号对比
# 原始频谱
ax3 = plt.subplot(gs[2, 0])
plt.plot(freqs_orig, mag_orig, 'b-', label='原始频谱', alpha=0.7)
plt.title('滤波前信号 (频域)')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅度')
plt.legend()
plt.grid(True)

# 滤波后频谱
ax4 = plt.subplot(gs[2, 1])
plt.plot(freqs_orig, mag_filt, 'r-', label='滤波后频谱', alpha=0.7)
plt.axvspan(lowcut, highcut, color='green', alpha=0.1, label='通带范围')
plt.title('滤波后信号 (频域)')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅度')
plt.legend()
plt.grid(True)

# 设置相同的y轴范围以便比较
mag_min = min(np.min(mag_orig), np.min(mag_filt))
mag_max = max(np.max(mag_orig), np.max(mag_filt))
ax3.set_ylim(mag_min, mag_max)
ax4.set_ylim(mag_min, mag_max)

# 7.4 绘制滤波前后信号对比（同一图表）
plt.subplot(gs[3, :])
plt.plot(freqs_orig, mag_orig, 'b-', label='原始频谱', alpha=0.7)
plt.plot(freqs_orig, mag_filt, 'r-', label='滤波后频谱', alpha=0.7)
plt.axvspan(lowcut, highcut, color='green', alpha=0.1, label='通带范围')
plt.title('滤波前后频谱对比')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅度')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 8. 分析滤波效果
# 检测滤波后信号的主要频率成分
peak_indices, _ = find_peaks(mag_filt, height=0.1*np.max(mag_filt), distance=10)
if len(peak_indices) > 0:
    main_freqs = freqs_orig[peak_indices]
    print("滤波后信号的主要频率成分:")
    for i, freq in enumerate(main_freqs):
        if lowcut <= freq <= highcut:
            print(f"  峰值 {i+1}: {freq:.2f} Hz (在通带内)")
        else:
            print(f"  峰值 {i+1}: {freq:.2f} Hz (在通带外)")


# 9. 保存滤波后的数据
# 保存时域数据
filtered_df = pd.DataFrame({
    'Time': time,
    'Original_Signal': data_detrended,
    'Filtered_Signal': filtered_data
})
filtered_df.to_csv(r'D:\python_project\data\filtered_signal_bandpass_2025-10-01_11-07-26.csv', index=False)

"""
# 保存频域数据
freq_df = pd.DataFrame({
    'Frequency': freqs_orig,
    'Original_Magnitude': mag_orig,
    'Filtered_Magnitude': mag_filt
})
freq_df.to_csv(r'D:\python_project\data\frequency_analysis_bandpass.csv', index=False)

print("滤波完成！数据已保存。")
"""
