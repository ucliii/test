import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, find_peaks, detrend, savgol_filter
from scipy.fft import fft, fftfreq
import matplotlib.gridspec as gridspec
import warnings
import pywt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

# 1. 读取CSV文件
file_path = r'D:\python_project\data\cleaned_data_2025-10-01_11-07-26.csv'
df = pd.read_csv(file_path, header=None, encoding='utf-8')  

# 2. 提取前12列数据
num_columns = 12
data_columns = df.iloc[:, :num_columns].values.T
print(f"数据形状: {data_columns.shape}")

# 3. 设置采样频率
sampling_rate = 100
print(f"使用的采样率: {sampling_rate} Hz")

# 4. 数据分段函数
def split_channel_data(channel_data):
    """将单个通道数据分成三段"""
    x = len(channel_data)
    segment1 = channel_data[0:x//3-180]
    segment2 = channel_data[x//3+90:2*x//3-90]
    segment3 = channel_data[2*x//3+180:x]
    return [segment1, segment2, segment3]

# 5. 智能脉搏波去噪函数
def intelligent_pulse_denoise(signal, fs=100, pulse_band=[0.5, 3.0]):
    """
    智能脉搏波去噪，专门处理与信号频带混叠的突发抽搐噪声
    """
    # 预处理
    signal_clean = preprocess_segment(signal)
    
    # 1. 脉搏波特征提取
    pulse_features = extract_pulse_features(signal_clean, fs)
    
    # 2. 噪声检测和分类
    noise_mask, noise_info = detect_and_classify_noise(signal_clean, pulse_features, fs)
    
    # 3. 基于特征的信号重建
    if np.sum(noise_mask) > 0:
        reconstructed_signal = reconstruct_signal(signal_clean, noise_mask, pulse_features, fs)
    else:
        reconstructed_signal = signal_clean
    
    # 4. 后处理平滑
    final_signal = postprocess_signal(reconstructed_signal, fs)
    
    return final_signal, noise_mask, noise_info

def extract_pulse_features(signal, fs):
    """提取脉搏波特征"""
    features = {}
    
    # 基本统计特征
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['dynamic_range'] = np.max(signal) - np.min(signal)
    
    # 时域特征
    features['gradient'] = np.gradient(signal)
    features['gradient_std'] = np.std(features['gradient'])
    
    # 检测可能的脉搏峰值
    peaks, peak_properties = find_peaks(signal, 
                                      height=np.mean(signal) + 0.5*np.std(signal),
                                      distance=int(0.5 * fs))  # 至少0.5秒间隔
    
    features['peaks'] = peaks
    features['peak_heights'] = peak_properties['peak_heights'] if len(peaks) > 0 else np.array([])
    
    # 计算脉搏周期特征
    if len(peaks) >= 3:
        peak_intervals = np.diff(peaks) / fs
        features['heart_rate'] = 60 / np.mean(peak_intervals)
        features['hrv'] = np.std(peak_intervals)  # 心率变异性
    else:
        features['heart_rate'] = None
        features['hrv'] = None
    
    # 频域特征
    freqs, spectrum = compute_spectrum(signal, fs)
    pulse_mask = (freqs >= 0.5) & (freqs <= 3.0)
    
    if np.any(pulse_mask):
        pulse_energy = np.sum(spectrum[pulse_mask])
        total_energy = np.sum(spectrum[freqs <= 10])
        features['pulse_energy_ratio'] = pulse_energy / total_energy
        features['dominant_freq'] = freqs[pulse_mask][np.argmax(spectrum[pulse_mask])]
    else:
        features['pulse_energy_ratio'] = 0
        features['dominant_freq'] = 0
    
    return features

def detect_and_classify_noise(signal, pulse_features, fs):
    """检测和分类噪声"""
    n = len(signal)
    noise_mask = np.zeros(n, dtype=bool)
    noise_info = {'burst_regions': [], 'baseline_drift': False, 'high_freq_noise': False}
    
    # 1. 检测突发抽搐噪声（基于梯度分析）
    gradient = np.gradient(signal)
    gradient_abs = np.abs(gradient)
    
    # 自适应梯度阈值
    grad_threshold = np.mean(gradient_abs) + 3 * np.std(gradient_abs)
    
    # 找到梯度突变的点
    burst_points = np.where(gradient_abs > grad_threshold)[0]
    
    if len(burst_points) > 0:
        # 扩展噪声区域（前后各0.1秒）
        expand_samples = int(0.1 * fs)
        for point in burst_points:
            start = max(0, point - expand_samples)
            end = min(n, point + expand_samples)
            noise_mask[start:end] = True
            
            noise_info['burst_regions'].append((start, end))
    
    # 2. 检测高频噪声（基于局部变异性）
    window_size = int(0.2 * fs)  # 200ms窗口
    local_variability = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - window_size//2)
        end = min(n, i + window_size//2)
        local_variability[i] = np.std(signal[start:end])
    
    # 检测异常高的局部变异性
    var_threshold = np.mean(local_variability) + 2 * np.std(local_variability)
    high_var_points = np.where(local_variability > var_threshold)[0]
    
    # 只标记那些不在突发噪声区域的高变异性点
    for point in high_var_points:
        if not noise_mask[point]:
            start = max(0, point - window_size//4)
            end = min(n, point + window_size//4)
            noise_mask[start:end] = True
    
    # 3. 基于脉搏周期一致性的检测
    if pulse_features['heart_rate'] is not None:
        expected_period = fs * 60 / pulse_features['heart_rate']  # 期望的脉搏周期（样本数）
        
        # 检测不符合脉搏周期的异常波动
        if len(pulse_features['peaks']) >= 3:
            valid_signal = identify_valid_pulse_waves(signal, pulse_features['peaks'], expected_period, fs)
            # 将不符合脉搏波形的区域标记为噪声
            noise_mask = noise_mask | (~valid_signal)
    
    noise_info['noise_ratio'] = np.sum(noise_mask) / n
    return noise_mask, noise_info

def identify_valid_pulse_waves(signal, peaks, expected_period, fs):
    """识别有效的脉搏波形"""
    n = len(signal)
    valid_mask = np.ones(n, dtype=bool)
    
    if len(peaks) < 3:
        return valid_mask
    
    # 分析脉搏波形的形态特征
    for i in range(1, len(peaks)-1):
        start = peaks[i-1]
        end = peaks[i+1]
        segment = signal[start:end]
        
        # 计算波形的对称性、上升时间等特征
        if len(segment) > 10:
            # 简单的形态检查：上升沿应该比下降沿陡峭
            peak_pos = peaks[i] - start
            if peak_pos > 0 and peak_pos < len(segment)-1:
                rise_segment = segment[:peak_pos]
                fall_segment = segment[peak_pos:]
                
                if len(rise_segment) > 0 and len(fall_segment) > 0:
                    rise_slope = np.mean(np.gradient(rise_segment))
                    fall_slope = np.mean(np.gradient(fall_segment))
                    
                    # 如果下降比上升更陡峭，可能是异常
                    if abs(fall_slope) > 2 * abs(rise_slope) and abs(fall_slope) > 0.5:
                        # 标记这个波形为可能的噪声
                        expansion = int(0.05 * fs)  # 扩展50ms
                        noise_start = max(0, peaks[i] - expansion)
                        noise_end = min(n, peaks[i] + expansion)
                        valid_mask[noise_start:noise_end] = False
    
    return valid_mask

def reconstruct_signal(signal, noise_mask, pulse_features, fs):
    """重建被噪声污染的信号"""
    n = len(signal)
    reconstructed = signal.copy()
    
    # 获取噪声区域的索引
    noise_indices = np.where(noise_mask)[0]
    
    if len(noise_indices) == 0:
        return signal
    
    # 将连续的噪声区域分组
    noise_regions = []
    current_region = [noise_indices[0]]
    
    for i in range(1, len(noise_indices)):
        if noise_indices[i] - noise_indices[i-1] <= int(0.1 * fs):  # 100ms内的连续点
            current_region.append(noise_indices[i])
        else:
            noise_regions.append(current_region)
            current_region = [noise_indices[i]]
    
    noise_regions.append(current_region)
    
    # 对每个噪声区域进行修复
    for region in noise_regions:
        if len(region) == 0:
            continue
            
        start_idx = max(0, region[0] - 1)
        end_idx = min(n-1, region[-1] + 1)
        
        # 小区域使用线性插值
        if len(region) < int(0.2 * fs):  # 小于200ms
            if start_idx > 0 and end_idx < n-1:
                reconstructed[region[0]:region[-1]+1] = np.linspace(
                    reconstructed[start_idx], reconstructed[end_idx], len(region))
        
        # 大区域使用基于脉搏模式的预测
        else:
            # 使用前后有效数据预测
            before_segment = get_clean_segment_before(signal, noise_mask, region[0], fs)
            after_segment = get_clean_segment_after(signal, noise_mask, region[-1], fs)
            
            if before_segment is not None and after_segment is not None:
                # 基于脉搏周期进行插值
                predicted_segment = predict_pulse_pattern(before_segment, after_segment, 
                                                         len(region), pulse_features, fs)
                reconstructed[region[0]:region[-1]+1] = predicted_segment
            else:
                # 回退到样条插值
                from scipy.interpolate import CubicSpline
                clean_indices = np.where(~noise_mask)[0]
                if len(clean_indices) >= 4:
                    cs = CubicSpline(clean_indices, signal[clean_indices])
                    reconstructed[region[0]:region[-1]+1] = cs(np.arange(region[0], region[-1]+1))
    
    return reconstructed

def get_clean_segment_before(signal, noise_mask, current_idx, fs):
    """获取当前索引之前的干净信号段"""
    lookback = int(1.5 * fs)  # 向前看1.5秒
    start = max(0, current_idx - lookback)
    segment = signal[start:current_idx]
    mask_segment = noise_mask[start:current_idx]
    
    # 找到最近的干净段（至少0.5秒）
    min_clean_length = int(0.5 * fs)
    for i in range(len(mask_segment)-min_clean_length, -1, -1):
        if not np.any(mask_segment[i:i+min_clean_length]):
            return segment[i:i+min_clean_length]
    
    return None

def get_clean_segment_after(signal, noise_mask, current_idx, fs):
    """获取当前索引之后的干净信号段"""
    lookahead = int(1.5 * fs)  # 向后看1.5秒
    end = min(len(signal), current_idx + lookahead)
    segment = signal[current_idx:end]
    mask_segment = noise_mask[current_idx:end]
    
    # 找到最近的干净段（至少0.5秒）
    min_clean_length = int(0.5 * fs)
    for i in range(0, len(mask_segment) - min_clean_length + 1):
        if not np.any(mask_segment[i:i+min_clean_length]):
            return segment[i:i+min_clean_length]
    
    return None

def predict_pulse_pattern(before_segment, after_segment, target_length, pulse_features, fs):
    """基于脉搏模式预测信号"""
    # 简单的加权平均插值
    if before_segment is None or after_segment is None:
        return np.zeros(target_length)
    
    # 使用前后段的特征进行插值
    combined_pattern = np.concatenate([before_segment, after_segment])
    
    # 如果知道心率，可以尝试基于周期进行预测
    if pulse_features['heart_rate'] is not None:
        expected_period = int(fs * 60 / pulse_features['heart_rate'])
        
        # 使用周期性的模式进行插值
        if len(combined_pattern) >= expected_period:
            # 提取一个完整的周期模式
            pattern = combined_pattern[-expected_period:] if len(after_segment) >= expected_period else combined_pattern[:expected_period]
            
            # 重复模式来填充目标长度
            repeats = target_length // len(pattern) + 1
            predicted = np.tile(pattern, repeats)[:target_length]
            return predicted
    
    # 回退到线性插值
    return np.linspace(before_segment[-1], after_segment[0], target_length)

def postprocess_signal(signal, fs):
    """后处理：轻度平滑"""
    # 使用Savitzky-Golay滤波器进行轻度平滑
    window_length = min(11, len(signal) // 10 * 2 + 1)  # 确保是奇数
    if window_length > 5:
        try:
            smoothed = savgol_filter(signal, window_length, 3)
            return smoothed
        except:
            return signal
    return signal

def compute_spectrum(signal, fs):
    """计算信号的频谱"""
    n = len(signal)
    if n == 0:
        return np.array([]), np.array([])
    
    window = np.hanning(n)
    windowed_signal = signal * window
    
    fft_result = fft(windowed_signal)
    freqs = fftfreq(n, d=1/fs)
    
    magnitude = 2 * np.abs(fft_result) / np.sum(window)
    
    # 只保留正频率部分
    positive_freq_idx = freqs >= 0
    return freqs[positive_freq_idx], magnitude[positive_freq_idx]

def preprocess_segment(segment_data):
    """数据预处理：去除直流分量和趋势"""
    if np.any(pd.isnull(segment_data)):
        segment_data = pd.Series(segment_data).fillna(method='ffill').values
    
    # 去除直流分量和线性趋势
    segment_detrended = detrend(segment_data - np.mean(segment_data))
    
    return segment_detrended

# 6. 选择要分析的通道和分段
channel_to_analyze = 8  # 选择第8通道
segment_to_plot = 0     # 选择第1个分段

channel_idx = channel_to_analyze - 1

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

# 使用智能去噪
denoised_signal, noise_mask, noise_info = intelligent_pulse_denoise(selected_segment, sampling_rate)

print(f"分段 {segment_to_plot+1} 处理完成")
print(f"检测到噪声比例: {noise_info['noise_ratio']*100:.2f}%")
print(f"突发噪声区域: {len(noise_info['burst_regions'])} 个")

# 7. 绘制结果
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 计算时间轴
time = np.arange(len(original_processed)) / sampling_rate

# 子图1: 原始信号和噪声检测
axes[0].plot(time, original_processed, 'b-', linewidth=1.2, label='原始信号', alpha=0.7)
axes[0].fill_between(time, original_processed.min(), original_processed.max(), 
                    where=noise_mask, color='red', alpha=0.3, label='检测到的噪声区域')
axes[0].set_title(f'通道 {channel_to_analyze} - 噪声检测结果')
axes[0].set_ylabel('幅度')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 子图2: 时域对比
axes[1].plot(time, original_processed, 'b-', linewidth=1.2, label='原始信号', alpha=0.7)
axes[1].plot(time, denoised_signal, 'r-', linewidth=1.5, label='智能去噪后')
axes[1].set_title('时域信号对比')
axes[1].set_ylabel('幅度')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 子图3: 频域对比
freqs_orig, mag_orig = compute_spectrum(original_processed, sampling_rate)
freqs_denoised, mag_denoised = compute_spectrum(denoised_signal, sampling_rate)

valid_range = (freqs_orig >= 0) & (freqs_orig <= 10)
axes[2].plot(freqs_orig[valid_range], mag_orig[valid_range], 'b-', linewidth=1.5, 
            label='原始频谱', alpha=0.7)
axes[2].plot(freqs_denoised[valid_range], mag_denoised[valid_range], 'r-', 
            linewidth=1.5, label='去噪后频谱')
axes[2].set_title('频域对比')
axes[2].set_xlabel('频率 (Hz)')
axes[2].set_ylabel('幅度')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 8. 性能评估
def calculate_snr_improvement(original, denoised):
    """计算信噪比改善"""
    noise = original - denoised
    signal_power = np.var(denoised)
    noise_power = np.var(noise)
    
    if noise_power > 0:
        return 10 * np.log10(signal_power / noise_power)
    else:
        return float('inf')

def calculate_energy_preservation(original, denoised, freq_band=[0.5, 3.0], fs=100):
    """计算主要频带能量保留"""
    freqs_orig, mag_orig = compute_spectrum(original, fs)
    freqs_denoised, mag_denoised = compute_spectrum(denoised, fs)
    
    band_mask = (freqs_orig >= freq_band[0]) & (freqs_orig <= freq_band[1])
    
    if np.any(band_mask):
        orig_energy = np.sum(mag_orig[band_mask]**2)
        denoised_energy = np.sum(mag_denoised[band_mask]**2)
        return denoised_energy / orig_energy * 100
    else:
        return 0

# 计算性能指标
snr_improvement = calculate_snr_improvement(original_processed, denoised_signal)
energy_preservation = calculate_energy_preservation(original_processed, denoised_signal)

print(f"\n智能去噪性能评估:")
print("=" * 50)
print(f"信噪比改善: {snr_improvement:.2f} dB")
print(f"主要频带能量保留: {energy_preservation:.2f}%")
print(f"处理的噪声区域: {len(noise_info['burst_regions'])} 个")

# 如果有脉搏率信息，显示出来
freqs, mag = compute_spectrum(denoised_signal, sampling_rate)
pulse_range = (freqs >= 0.8) & (freqs <= 2.5)
if np.any(pulse_range):
    pulse_freq = freqs[pulse_range][np.argmax(mag[pulse_range])]
    pulse_bpm = pulse_freq * 60
    print(f"估计脉搏率: {pulse_bpm:.1f} BPM")