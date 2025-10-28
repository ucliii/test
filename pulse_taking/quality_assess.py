import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, find_peaks, detrend
import matplotlib.gridspec as gridspec
import warnings
from functools import cmp_to_key

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

# 4. 设计巴特沃斯带通滤波器 (0.6-4Hz)
def butter_bandpass(lowcut, highcut, fs, order=5):
    """设计巴特沃斯带通滤波器"""
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyq  # 归一化低频截止频率
    high = highcut / nyq  # 归一化高频截止频率
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """应用巴特沃斯带通滤波器"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# 设置滤波器参数
lowcut = 0.6  # 低频截止频率 (Hz)
highcut = 4.0  # 高频截止频率 (Hz)
order = 5  # 滤波器阶数

# 5. 数据预处理和滤波处理（针对单个分段）
def process_single_segment(segment_data, fs, lowcut, highcut, order):
    """处理单个数据分段"""
    # 数据预处理
    if np.any(pd.isnull(segment_data)):
        segment_data = pd.Series(segment_data).fillna(method='ffill').values
    
    # 去除直流分量和线性趋势
    segment_detrended = detrend(segment_data - np.mean(segment_data))
    
    # 应用带通滤波器
    filtered_segment = butter_bandpass_filter(segment_detrended, lowcut, highcut, fs, order)
    
    return segment_detrended, filtered_segment

# 6. 计算频谱函数
def compute_spectrum(signal, fs):
    """计算信号的频谱"""
    window = np.hanning(len(signal))
    windowed_signal = signal * window
    window_sum = np.sum(window)
    
    fft_result = np.fft.fft(windowed_signal)
    freqs = np.fft.fftfreq(len(fft_result), d=1/fs)
    
    magnitude = np.abs(fft_result) / (window_sum / 2)
    magnitude[0] = magnitude[0] / 2  # 直流分量校正
    
    return freqs, magnitude

# 7. 基于相邻频段比较的多峰检测方法（改进版）
def detect_spectral_peaks_by_bands(freqs, magnitude, low_freq=0.6, high_freq=4.0, band_width=0.2):
    """
    改进的基于相邻频段比较的多峰检测方法
    不再使用整个频谱的能量均值作为阈值，而是相邻频段间进行比较
    """
    # 提取有效频带
    mask = (freqs >= low_freq) & (freqs <= high_freq)
    effective_freqs = freqs[mask]
    effective_magnitude = magnitude[mask]
    
    if len(effective_freqs) == 0:
        return [], [], []
    
    # 第一步：将有效频带划分为多个小频段
    band_starts = np.arange(low_freq, high_freq, band_width)
    band_centers = band_starts + band_width / 2
    band_energies = []
    
    for start in band_starts:
        end = start + band_width
        if end > high_freq:
            end = high_freq
        
        # 计算该频段内的能量
        band_mask = (effective_freqs >= start) & (effective_freqs < end)
        if np.any(band_mask):
            band_energy = np.trapz(effective_magnitude[band_mask]**2, effective_freqs[band_mask])
        else:
            band_energy = 0
        
        band_energies.append(band_energy)
    
    band_energies = np.array(band_energies)
    n_bands = len(band_energies)
    
    # 第二步：标记候选峰值频段（比相邻频段能量高）
    candidate_peaks = []
    
    for i in range(n_bands):
        current_energy = band_energies[i]
        
        # 检查左相邻频段（如果存在）
        left_ok = True
        if i > 0:
            left_energy = band_energies[i-1]
            left_ok = current_energy > left_energy
        
        # 检查右相邻频段（如果存在）
        right_ok = True
        if i < n_bands - 1:
            right_energy = band_energies[i+1]
            right_ok = current_energy > right_energy
        
        # 如果当前频段能量高于左右相邻频段，则标记为候选峰值
        if left_ok and right_ok:
            candidate_peaks.append({
                'index': i,
                'center_freq': band_centers[i],
                'start_freq': band_starts[i],
                'end_freq': min(band_starts[i] + band_width, high_freq),
                'energy': current_energy
            })
    
    # 第三步：合并距离过近的候选峰值（改进版）
    # 首先按中心频率排序
    candidate_peaks.sort(key=lambda x: x['center_freq'])
    
    # 使用标记数组来跟踪哪些峰值被保留
    keep_peak = [True] * len(candidate_peaks)
    
    # 检查所有峰值对的距离
    for i in range(len(candidate_peaks)):
        if not keep_peak[i]:
            continue
            
        for j in range(i+1, len(candidate_peaks)):
            if not keep_peak[j]:
                continue
                
            # 计算两个峰值之间的距离
            distance = candidate_peaks[j]['center_freq'] - candidate_peaks[i]['center_freq']
            
            # 如果距离小于3个单位频段，则合并它们
            if distance < 3 * band_width:
                # 保留能量较大的峰值，标记能量较小的为不保留
                if candidate_peaks[i]['energy'] >= candidate_peaks[j]['energy']:
                    keep_peak[j] = False
                else:
                    keep_peak[i] = False
                    break  # 当前峰值已被标记为不保留，跳出内层循环
    
    # 收集最终保留的峰值
    final_peaks = []
    for i, keep in enumerate(keep_peak):
        if keep:
            final_peaks.append(candidate_peaks[i])
    
    # 转换为与之前兼容的输出格式
    peak_bands = []
    for peak in final_peaks:
        peak_bands.append((peak['start_freq'], peak['end_freq'], peak['energy']))
    
    return peak_bands, band_energies, band_centers

def calculate_absolute_energy(freqs, magnitude, low_freq=0.6, high_freq=4.0):
    """
    计算有效频谱的绝对能量
    """
    # 计算总能量
    mask = (freqs >= low_freq) & (freqs <= high_freq)
    effective_freqs = freqs[mask]
    effective_magnitude = magnitude[mask]
    total_energy = np.trapz(effective_magnitude**2, effective_freqs)
    
    # 检测能量集中频段
    peak_bands, _, _ = detect_spectral_peaks_by_bands(freqs, magnitude, low_freq, high_freq)
    peak_count = len(peak_bands)
    
    # 计算能量集中频段的总能量
    peak_band_energy = sum([energy for _, _, energy in peak_bands])
    
    if total_energy == 0:
        energy_concentration = 0
    else:
        energy_concentration = peak_band_energy / total_energy
    
    return total_energy, energy_concentration, peak_count, peak_bands

def calculate_peak_count_score(peak_count):
    """计算波峰数得分：3-4个最佳，2个次之，其他最差"""
    if peak_count in [3, 4]:
        return 1.0  # 最佳
    elif peak_count == 2:
        return 0.7  # 次之
    else:
        return 0.4  # 最差

def normalize_energy_sqrt(energy_value):
    """归一化能量值到0-1范围（使用平方根变换）"""
    if energy_value <= 0:
        return 0
    
    # 使用平方根变换，然后归一化到0-1范围
    # 假设最大能量值为1000，这样443.24的平方根归一化约为0.67
    sqrt_energy = np.sqrt(energy_value)
    max_sqrt = np.sqrt(1000)  # 假设最大能量为1000
    normalized = sqrt_energy / max_sqrt
    return min(1.0, normalized)

# 8. 信号质量评估函数（基于三个指标）
def analyze_signal_quality(segment_data, fs=100):
    """分析信号质量 - 基于波峰数、能量集中度和绝对能量"""
    # 对分段数据进行预处理和滤波
    segment_detrended, filtered_segment = process_single_segment(segment_data, fs, lowcut, highcut, order)
    
    # 计算滤波后信号的频谱
    freqs_filt, mag_filt = compute_spectrum(filtered_segment, fs)
    
    # 计算三个指标
    absolute_energy, energy_concentration, peak_count, peak_bands = calculate_absolute_energy(freqs_filt, mag_filt)
    
    # 计算各项得分
    peak_score = calculate_peak_count_score(peak_count)  # 波峰数得分
    energy_conc_score = energy_concentration  # 能量集中度得分（已经是0-1）
    absolute_energy_score = normalize_energy_sqrt(absolute_energy)  # 绝对能量得分（使用平方根归一化）
    
    # 综合评分：波峰数30% + 能量集中度40% + 绝对能量30%
    combined_score = (peak_score * 0.3 + energy_conc_score * 0.4 + absolute_energy_score * 0.3) * 5
    
    return combined_score, peak_score, energy_conc_score, absolute_energy_score, peak_count, absolute_energy, peak_bands

# 9. 数据分段函数
def split_channel_data(channel_data):
    """将单个通道数据分成三段"""
    x = len(channel_data)
    
    # 计算分段索引
    segment1 = channel_data[0:x//3-180]
    segment2 = channel_data[x//3+90:2*x//3-90]
    segment3 = channel_data[2*x//3+180:x]
    
    return [segment1, segment2, segment3]

# 10. 通道比较函数（按照新策略）
def compare_channels(channel_segment_scores):
    """按照新策略比较通道评分"""
    n_channels = len(channel_segment_scores)
    
    # 为每个通道存储三个分段的评分（按从高到低排序）
    sorted_segment_scores = []
    for i in range(n_channels):
        # 对每个通道的三个分段评分进行排序（从高到低）
        sorted_scores = sorted(channel_segment_scores[i], reverse=True)
        sorted_segment_scores.append(sorted_scores)
    
    # 创建索引列表
    indices = list(range(n_channels))
    
    def compare_channels_func(i, j):
        # 获取两个通道的排序后分段评分
        scores_i = sorted_segment_scores[i]
        scores_j = sorted_segment_scores[j]
        
        # 比较第一高分
        if abs(scores_i[0] - scores_j[0]) / max(scores_i[0], scores_j[0]) > 0.1:
            return scores_j[0] - scores_i[0]  # 降序排列
        
        # 比较第二高分
        if abs(scores_i[1] - scores_j[1]) / max(scores_i[1], scores_j[1]) > 0.1:
            return scores_j[1] - scores_i[1]
        
        # 比较第三高分
        if abs(scores_i[2] - scores_j[2]) / max(scores_i[2], scores_j[2]) > 0.1:
            return scores_j[2] - scores_i[2]
        
        # 如果所有分段差距都小于10%，则以第一高分比较结果为最终结果
        return scores_j[0] - scores_i[0]
    
    # 排序
    ranked_indices = sorted(indices, key=cmp_to_key(compare_channels_func))
    
    return ranked_indices

# 11. 可视化单个通道的三个分段
def visualize_single_channel(raw_data, channel_scores, channel_idx, sampling_rate=100):
    """
    可视化单个通道的三个分段
    
    参数:
    - raw_data: 原始数据
    - channel_scores: 通道评分信息
    - channel_idx: 通道的原始索引 (0-based)
    - sampling_rate: 采样频率
    """
    # 获取通道的原始数据
    channel_raw = raw_data[channel_idx]
    
    # 分割通道数据
    segments = split_channel_data(channel_raw)
    
    # 创建图形
    plt.figure(figsize=(14, 8))
    
    # 对每个分段进行处理和可视化
    for seg_idx, segment in enumerate(segments):
        # 处理分段数据
        segment_detrended, segment_filtered = process_single_segment(segment, sampling_rate, lowcut, highcut, order)
        
        # 计算频谱
        segment_freqs, segment_mag = compute_spectrum(segment_filtered, sampling_rate)
        
        # 检测能量集中频段
        segment_peak_bands, _, _ = detect_spectral_peaks_by_bands(segment_freqs, segment_mag)
        
        # 时域图 - 第一列
        plt.subplot(3, 2, seg_idx*2 + 1)
        time = np.arange(len(segment_detrended)) / sampling_rate
        plt.plot(time, segment_detrended, 'b-', alpha=0.8, linewidth=1)
        plt.title(f'通道 {channel_idx+1} - 分段 {seg_idx+1} 时域信号\n评分: {channel_scores[channel_idx][seg_idx]:.2f}')
        plt.xlabel('时间 (秒)')
        plt.ylabel('幅度')
        plt.grid(True, alpha=0.3)
        
        # 频域图 - 第二列
        plt.subplot(3, 2, seg_idx*2 + 2)
        plt.plot(segment_freqs[segment_freqs >= 0], segment_mag[segment_freqs >= 0], 'g-', alpha=0.8, linewidth=1.5)
        
        # 标记检测到的能量集中频段
        for band_start, band_end, energy in segment_peak_bands:
            plt.axvspan(band_start, band_end, color='red', alpha=0.3)
        
        plt.axvspan(lowcut, highcut, color='green', alpha=0.2, label='有效频带')
        plt.axvline(lowcut, color='k', linestyle='--', alpha=0.7)
        plt.axvline(highcut, color='k', linestyle='--', alpha=0.7)
        
        plt.title(f'通道 {channel_idx+1} - 分段 {seg_idx+1} 频谱\n检测到 {len(segment_peak_bands)} 个能量集中频段')
        plt.xlabel('频率 (Hz)')
        plt.ylabel('幅度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 10)
    
    plt.tight_layout()
    plt.show()

# 12. 主函数
def main():
    """主函数"""
    print("开始脉搏信号质量评估...")
    print("=" * 50)
    
    # 批量质量评估
    print("\n正在进行质量评估...")
    channel_segment_scores = []  # 存储每个通道的三个分段评分
    channel_segment_details = []  # 存储每个通道的详细分段信息
    
    for channel_idx in range(num_columns):
        # 获取原始通道数据
        original_data = data_columns[channel_idx]
        
        # 分割通道数据
        segments = split_channel_data(original_data)
        
        # 评估每个分段的质量
        segment_scores = []
        segment_details = []
        
        for segment in segments:
            # 对每个分段分别进行质量评估
            quality_score, peak_score, energy_conc_score, abs_energy_score, peak_count, absolute_energy, peak_bands = analyze_signal_quality(segment)
            segment_scores.append(quality_score)
            segment_details.append({
                'quality_score': quality_score,
                'peak_score': peak_score,
                'energy_conc_score': energy_conc_score,
                'abs_energy_score': abs_energy_score,
                'peak_count': peak_count,
                'absolute_energy': absolute_energy,
                'peak_bands': peak_bands
            })
        
        channel_segment_scores.append(segment_scores)
        channel_segment_details.append({
            'channel': channel_idx + 1,
            'segment_scores': segment_scores,
            'segment_details': segment_details
        })
        print(f"通道 {channel_idx+1} 评估完成")
    
    # 按照新策略进行通道排序
    ranked_indices = compare_channels(channel_segment_scores)
    ranked_results = [channel_segment_details[i] for i in ranked_indices]
    
    # 输出评估结果（精简版）
    print("\n" + "="*80)
    print("脉搏信号质量评估报告")
    print("="*80)
    print("排名 | 通道 | 分段1(峰/集/能) | 分段2(峰/集/能) | 分段3(峰/集/能)")
    print("-"*80)
    
    for i, res in enumerate(ranked_results):
        seg1_detail = res['segment_details'][0]
        seg2_detail = res['segment_details'][1]
        seg3_detail = res['segment_details'][2]
        
        print(f"{i+1:2d}   |  {res['channel']:2d}  | "
              f"{seg1_detail['peak_count']:1d}/{seg1_detail['energy_conc_score']:.2f}/{seg1_detail['absolute_energy']:6.2f} | "
              f"{seg2_detail['peak_count']:1d}/{seg2_detail['energy_conc_score']:.2f}/{seg2_detail['absolute_energy']:6.2f} | "
              f"{seg3_detail['peak_count']:1d}/{seg3_detail['energy_conc_score']:.2f}/{seg3_detail['absolute_energy']:6.2f}")
    
    # 可视化单个通道的三个分段
    # 在这里修改要显示的通道号 (1-12)
    channel_to_show = 11  # 要显示的通道 (原始通道号)
    
    if 1 <= channel_to_show <= num_columns:
        print(f"\n显示通道 {channel_to_show} 的三个分段分析...")
        visualize_single_channel(data_columns, channel_segment_scores, 
                               channel_to_show-1, sampling_rate)
    else:
        print("指定的通道号无效")
    
    print(f"\n评估完成！共评估了{num_columns}个通道的信号质量。")

if __name__ == "__main__":
    main()