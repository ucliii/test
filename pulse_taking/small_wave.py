import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, find_peaks, detrend
import matplotlib.gridspec as gridspec
import warnings
import pywt

# 设置中文字体支持，解决警告问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
warnings.filterwarnings("ignore")  # 忽略警告信息

# 1. 读取CSV文件
file_path = r'D:\python_project\data\cleaned_data_2025-10-01_11-07-26.csv'
df = pd.read_csv(file_path, header=None, encoding='utf-8')  

# 2. 提取前12列数据
num_columns = 12  # 处理前12列
data_columns = df.iloc[:, :num_columns].values.T  # 转置，使每行代表一列数据
print(f"数据形状: {data_columns.shape} (列数×样本数)")

# 3. 设置采样频率
sampling_rate = 100  # Hz
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

# 5. 小波变换去噪函数
def wavelet_denoise(signal, wavelet='sym8', level=None, threshold_type='soft'):
    """
    小波变换去噪函数
    """
    # 自动计算合适的分解层数
    if level is None:
        max_level = pywt.dwt_max_level(len(signal), wavelet)
        desired_level = int(np.log2(sampling_rate / (2 * 3)))
        level = min(max_level, desired_level, 6)
    
    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # 阈值策略：尺度相关阈值
    thresholded_coeffs = [coeffs[0]]
    
    for i in range(1, len(coeffs)):
        sigma = np.median(np.abs(coeffs[i])) / 0.6745
        scale_factor = 1.5 - (i-1) * 0.2
        scale_factor = max(0.8, min(1.5, scale_factor))
        threshold = scale_factor * sigma * np.sqrt(2 * np.log(len(coeffs[i])))
        
        if threshold_type == 'soft':
            detail_coeff = pywt.threshold(coeffs[i], threshold, mode='soft')
        else:
            detail_coeff = pywt.threshold(coeffs[i], threshold, mode='hard')
        
        thresholded_coeffs.append(detail_coeff)
    
    # 小波重构
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet)
    
    # 确保输出长度一致
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    elif len(denoised_signal) < len(signal):
        denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)))
    
    return denoised_signal

# 6. 数据预处理
def preprocess_segment(segment_data):
    """数据预处理：去除直流分量和趋势"""
    if np.any(pd.isnull(segment_data)):
        segment_data = pd.Series(segment_data).fillna(method='ffill').values
    
    # 去除直流分量和线性趋势
    segment_detrended = detrend(segment_data - np.mean(segment_data))
    
    return segment_detrended

# 7. 计算频谱函数
def compute_spectrum(signal, fs):
    """计算信号的频谱"""
    window = np.hanning(len(signal))
    windowed_signal = signal * window
    window_sum = np.sum(window)
    
    fft_result = np.fft.fft(windowed_signal)
    freqs = np.fft.fftfreq(len(fft_result), d=1/fs)
    
    magnitude = np.abs(fft_result) / (window_sum / 2)
    magnitude[0] = magnitude[0] / 2
    
    return freqs, magnitude

# 8. 定量评估函数
def calculate_snr(original, filtered):
    """计算信噪比改善"""
    signal_power = np.var(filtered)
    noise_power = np.var(original - filtered)
    if noise_power > 0:
        return 10 * np.log10(signal_power / noise_power)
    else:
        return float('inf')

def calculate_rmse(original, filtered):
    """计算均方根误差"""
    return np.sqrt(np.mean((original - filtered)**2))

# 9. 选择要分析的通道和分段
channel_to_analyze = 8  # 选择第8通道
segment_to_plot = 0     # 选择第1个分段

channel_idx = channel_to_analyze - 1  # 转换为0-based索引

if channel_idx < 0 or channel_idx >= num_columns:
    print(f"错误：通道号 {channel_to_analyze} 超出范围 (1-{num_columns})")
    exit()

print(f"分析通道 {channel_to_analyze}...")

# 获取通道数据并分段
channel_data = data_columns[channel_idx]
segments = split_channel_data(channel_data)

# 处理选定的分段
selected_segment = segments[segment_to_plot]
original_processed = preprocess_segment(selected_segment)
wavelet_denoised = wavelet_denoise(original_processed)

print(f"分段 {segment_to_plot+1} 处理完成")

# 10. 绘制两张子图：时域对比和频域对比
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# 计算时间轴和频谱
time = np.arange(len(original_processed)) / sampling_rate
freqs_orig, mag_orig = compute_spectrum(original_processed, sampling_rate)
freqs_wavelet, mag_wavelet = compute_spectrum(wavelet_denoised, sampling_rate)

# 只关注有意义的频率范围 (0-10Hz)
valid_range = (freqs_orig >= 0) & (freqs_orig <= 10)
freqs_display = freqs_orig[valid_range]
mag_orig_display = mag_orig[valid_range]
mag_wavelet_display = mag_wavelet[valid_range]

# 子图1: 时域对比
ax1.plot(time, original_processed, 'b-', linewidth=1.2, label='原始信号')
ax1.plot(time, wavelet_denoised, 'r-', linewidth=1.2, label='小波去噪后信号')
ax1.set_title(f'通道 {channel_to_analyze} - 分段 {segment_to_plot+1} 时域对比')
ax1.set_xlabel('时间 (秒)')
ax1.set_ylabel('幅度')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, time[-1])

# 子图2: 频域对比
ax2.plot(freqs_display, mag_orig_display, 'b-', linewidth=1.5, label='原始频谱')
ax2.plot(freqs_display, mag_wavelet_display, 'r-', linewidth=1.5, label='小波去噪后频谱')
ax2.set_title(f'通道 {channel_to_analyze} - 分段 {segment_to_plot+1} 频域对比')
ax2.set_xlabel('频率 (Hz)')
ax2.set_ylabel('幅度')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 10)

plt.tight_layout()
plt.show()

# 11. 定量分析
print(f"\n通道 {channel_to_analyze} - 分段 {segment_to_plot+1} 小波去噪效果分析:")
print("=" * 60)

# 计算0.5-5Hz频带的能量变化（主要信号频带）
signal_freq_range = (freqs_orig >= 0.5) & (freqs_orig <= 5)
signal_energy_orig = np.sum(mag_orig[signal_freq_range]**2)
signal_energy_wavelet = np.sum(mag_wavelet[signal_freq_range]**2)
signal_preservation = (signal_energy_wavelet / signal_energy_orig) * 100

# 定量评估
snr_improvement = calculate_snr(original_processed, wavelet_denoised)
rmse_value = calculate_rmse(original_processed, wavelet_denoised)

print(f"主要信号保护效果 (0.5-5Hz):")
print(f"  原始信号主要频带能量: {signal_energy_orig:.6f}")
print(f"  去噪后主要频带能量: {signal_energy_wavelet:.6f}")
print(f"  信号能量保留: {signal_preservation:.2f}%")

print(f"\n整体性能指标:")
print(f"  信噪比改善: {snr_improvement:.2f} dB")
print(f"  均方根误差: {rmse_value:.6f}")

print(f"\n处理完成！展示了通道 {channel_to_analyze} 第 {segment_to_plot+1} 分段的时频域对比图。")