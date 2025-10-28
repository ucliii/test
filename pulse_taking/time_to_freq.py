import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据分段函数（参考第一个代码）
def split_channel_data(channel_data):
    """将单个通道数据分成三段"""
    x = len(channel_data)
    
    # 计算分段索引
    segment1 = channel_data[0:x//3-180]
    segment2 = channel_data[x//3+90:2*x//3-90]
    segment3 = channel_data[2*x//3+180:x]
    
    return [segment1, segment2, segment3]

# 2. 计算频谱函数
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

# 3. 读取CSV文件
file_path = r'D:\python_project\data\ReceivedTofile-COM3-2025-10-21_10-51-41.DAT'
df = pd.read_csv(file_path, header=None, encoding='utf-8')  

# 4. 提取指定列数据
column_index = 8-1  # 第8列（索引从0开始）
data = df.iloc[:, column_index].values

print(f"数据长度: {len(data)} 个样本点")

# 5. 数据预处理
if np.any(pd.isnull(data)):
    print("数据中存在缺失值，正在填充...")
    data = pd.Series(data).fillna(method='ffill').values

# 6. 使用分段函数
segments = split_channel_data(data)
print(f"数据已分成 {len(segments)} 段")

# 7. 选择要处理的分段（这里选择第一段）
selected_segment_index = 0  # 可以选择0,1,2
segment_data = segments[selected_segment_index]

# 8. 对选定的分段进行去直流分量处理
segment_detrended = detrend(segment_data - np.mean(segment_data))

# 9. 设置采样频率
sampling_rate = 100  # Hz
print(f"使用的采样率: {sampling_rate} Hz")

# 10. 计算频谱
freqs, magnitude = compute_spectrum(segment_detrended, sampling_rate)

# 只关注有意义的频率范围 (0-10Hz)
valid_range = (freqs >= 0) & (freqs <= 10)
freqs_valid = freqs[valid_range]
magnitude_valid = magnitude[valid_range]

# 11. 绘制图形 - 只有两张图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# 时域图 - 经过去直流分量处理的波形
time_seg = np.arange(len(segment_detrended)) / sampling_rate
ax1.plot(time_seg, segment_detrended, 'b-', linewidth=1)
ax1.set_title(f'第{column_index+1}列 - 分段{selected_segment_index+1} 时域波形\n(去直流分量后)')
ax1.set_xlabel('时间 (秒)')
ax1.set_ylabel('幅度')
ax1.grid(True, alpha=0.3)

# 频域图 - 添加频带高亮区分
ax2.plot(freqs_valid, magnitude_valid, 'k-', linewidth=1.5, label='频谱')

# 添加频带高亮
ax2.axvspan(0, 0.5, color='lightgreen', alpha=0.3, label='0-0.5 Hz')
ax2.axvspan(0.5, 6, color='lightblue', alpha=0.3, label='0.5-6 Hz')
ax2.axvspan(6, 10, color='lightcoral', alpha=0.3, label='>6 Hz')

# 添加频带分隔线
ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
ax2.axvline(6, color='gray', linestyle='--', alpha=0.7, linewidth=1)

ax2.set_title(f'第{column_index+1}列 - 分段{selected_segment_index+1} 频域分析')
ax2.set_xlabel('频率 (Hz)')
ax2.set_ylabel('幅度')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 10)
ax2.legend()

plt.tight_layout()
plt.show()

# 12. 频谱分析
print(f"\n=== 第{column_index+1}列 - 分段{selected_segment_index+1} 频谱分析结果 ===")

# 找到最大幅度的频率
max_amp_idx = np.argmax(magnitude_valid)
dominant_freq = freqs_valid[max_amp_idx]
print(f"主导频率: {dominant_freq:.2f} Hz")
print(f"对应脉搏率: {dominant_freq * 60:.1f} BPM")

# 检测主要频率峰值
peak_indices, _ = find_peaks(magnitude_valid, height=0.1*np.max(magnitude_valid), distance=10)
if len(peak_indices) > 0:
    main_freqs = freqs_valid[peak_indices]
    print(f"主要频率成分:")
    for j, freq in enumerate(main_freqs):
        # 判断峰值所在的频带
        if freq <= 0.5:
            band = "0-0.5 Hz (低频)"
        elif freq <= 6:
            band = "0.5-6 Hz (中频)"
        else:
            band = ">6 Hz (高频)"
        print(f"  峰值 {j+1}: {freq:.2f} Hz - {band}")