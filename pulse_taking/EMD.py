import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import  detrend

import warnings

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

# 1. 读取CSV文件
file_path = r'D:\python_project\data\cleaned_data_2025_10_10_17_44_00.csv'
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

# 5. EEMD分解函数
def perform_eemd_decomposition(signal):
    """
    使用EEMD对信号进行分解
    """
    try:
        from PyEMD import EEMD
    except ImportError:
        print("请先安装PyEMD库: pip install EMD-signal")
        return None, None
    
    # 预处理：去趋势和去除直流分量
    signal_processed = detrend(signal)
    signal_processed = signal_processed - np.mean(signal_processed)
    
    # 创建EEMD对象
    eemd = EEMD(
        spline_kind='cubic',
        nbsym=4,
        DT=1.0/sampling_rate,
        max_imf=10,
        noise_width=0.05,
        ensemble_size=100,
    )
    
    try:
        # 执行EEMD分解
        imfs = eemd(signal_processed)
        print(f"EEMD分解得到 {len(imfs)} 个IMF")
        return imfs, signal_processed
        
    except Exception as e:
        print(f"EEMD分解失败: {e}")
        return None, None

# 6. 频谱计算函数
def compute_spectrum(signal, fs):
    """计算信号的频谱"""
    n = len(signal)
    window = np.hanning(n)
    windowed_signal = signal * window
    
    fft_result = np.fft.fft(windowed_signal, n)
    freqs = np.fft.fftfreq(n, d=1/fs)
    
    magnitude = 2 * np.abs(fft_result) / np.sum(window)
    
    # 只保留正频率部分
    positive_freq_idx = freqs >= 0
    return freqs[positive_freq_idx], magnitude[positive_freq_idx]

# 7. 选择要分析的通道
plot_column_index = 6-1
original_column_data = data_columns[plot_column_index]
segments = split_channel_data(original_column_data)
segment = segments[0]#分段

# 执行EEMD分解
imfs, signal_processed = perform_eemd_decomposition(segment)

if imfs is not None and len(imfs) > 0:
    print(f"EEMD分解完成，得到 {len(imfs)} 个IMF分量")
    
    # 计算时间轴
    time_seg = np.arange(len(signal_processed)) / sampling_rate
    
    # ==================== 按照要求的处理策略 ====================
    print("\n" + "="*60)
    print("处理策略：第一个分量和后4个分量直接去除，其他分量保持不变")
    print("="*60)
    
    processed_imfs = []
    processing_log = []
    
    for i, imf in enumerate(imfs):
        # 计算IMF的能量和主要频率特征
        imf_energy = np.sum(imf**2)
        freqs_imf, spec_imf = compute_spectrum(imf, sampling_rate)
        
        # 找到主要频率成分
        valid_freq_range = (freqs_imf >= 0.5) & (freqs_imf <= 10)
        if np.any(valid_freq_range):
            main_freq_idx = np.argmax(spec_imf[valid_freq_range])
            main_freq = freqs_imf[valid_freq_range][main_freq_idx]
        else:
            main_freq = 0
        
        print(f"IMF {i+1}: 能量={imf_energy:.2f}, 主频={main_freq:.2f}Hz -> ", end="")
        
        if i <= 1:  # 第一个分量：直接去除
            processed_imf = np.zeros_like(imf)
            processing_log.append(f"IMF1: 直接去除")
            
        elif i >= len(imfs) - 4:  # 最后4个分量：直接去除
            processed_imf = np.zeros_like(imf)
            processing_log.append(f"IMF{i+1}: 直接去除")
            
        else:  # 其他分量：保持不变
            processed_imf = imf
            processing_log.append(f"IMF{i+1}: 保持不变")
        
        processed_imfs.append(processed_imf)
        print(processing_log[-1])
    
    # 重构处理后的信号
    final_reconstructed = np.sum(processed_imfs, axis=0)
    
    # ==================== 第一幅图：IMF分量展示 ====================
    n_imfs = len(imfs)
    fig1, axes1 = plt.subplots(n_imfs, 2, figsize=(14, 9))
    
    # 如果只有一个IMF，将axes1转换为2D数组
    if n_imfs == 1:
        axes1 = np.array([axes1])
    
    for i in range(n_imfs):
        # 左列：时域图
        axes1[i, 0].plot(time_seg, imfs[i], 'b-', linewidth=1.2)
        axes1[i, 0].set_ylabel(f'IMF{i+1}')
        axes1[i, 0].grid(True, alpha=0.3)
        if i == n_imfs - 1:
            axes1[i, 0].set_xlabel('时间 (秒)')
        
        # 右列：频域图
        freqs_imf, spec_imf = compute_spectrum(imfs[i], sampling_rate)
        valid_range = freqs_imf <= 10
        axes1[i, 1].plot(freqs_imf[valid_range], spec_imf[valid_range], 'r-', linewidth=1.2)
        axes1[i, 1].set_ylabel('幅度')
        axes1[i, 1].grid(True, alpha=0.3)
        axes1[i, 1].set_xlim(0, 10)
        if i == n_imfs - 1:
            axes1[i, 1].set_xlabel('频率 (Hz)')
    
    # 设置第一幅图的标题
    axes1[0, 0].set_title('IMF分量时域图')
    axes1[0, 1].set_title('IMF分量频域图')
    
    plt.tight_layout()
    plt.show()
    
    # ==================== 第二幅图：处理结果对比（4x1布局） ====================
    fig2, axes2 = plt.subplots(4, 1, figsize=(12, 8))
    
    # 子图1: 原始信号时域
    axes2[0].plot(time_seg, signal_processed, 'b-', linewidth=1.5, label='原始信号')
    axes2[0].set_title('原始信号时域波形 (去除直流分量后)')
    axes2[0].set_ylabel('幅度')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    # 子图2: 处理后信号时域
    axes2[1].plot(time_seg, final_reconstructed, 'r-', linewidth=1.5, label='处理后信号')
    axes2[1].set_title('去除前2个和后4个IMF后的信号')
    axes2[1].set_ylabel('幅度')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    
    # 子图3: 原始信号频域
    freqs_orig, spec_orig = compute_spectrum(signal_processed, sampling_rate)
    valid_range = freqs_orig <= 10
    axes2[2].plot(freqs_orig[valid_range], spec_orig[valid_range], 'b-', linewidth=1.5, label='原始频谱')
    axes2[2].set_title('原始信号频域分析')
    axes2[2].set_ylabel('幅度')
    axes2[2].legend()
    axes2[2].grid(True, alpha=0.3)
    
    # 子图4: 处理后信号频域
    freqs_processed, spec_processed = compute_spectrum(final_reconstructed, sampling_rate)
    valid_range = freqs_processed <= 10
    axes2[3].plot(freqs_processed[valid_range], spec_processed[valid_range], 'r-', linewidth=1.5, label='处理后频谱')
    axes2[3].set_title('处理后信号频域分析')
    axes2[3].set_xlabel('频率 (Hz)')
    axes2[3].set_ylabel('幅度')
    axes2[3].legend()
    axes2[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ==================== 性能评估 ====================
    # 计算重构误差
    reconstruction_error = np.mean((signal_processed - final_reconstructed)**2)
    
    # 计算信号质量指标
    max_amplitude = np.max(np.abs(signal_processed))
    relative_mse = reconstruction_error / (max_amplitude**2) * 100
    
    print(f"\n处理性能评估:")
    print(f"重构误差 (MSE): {reconstruction_error:.6f}")
    print(f"信号最大幅值: {max_amplitude:.2f}")
    print(f"相对MSE: {relative_mse:.4f}%")
    
    # 计算处理前后信号能量
    original_energy = np.sum(signal_processed**2)
    processed_energy = np.sum(final_reconstructed**2)
    energy_reduction = (1 - processed_energy/original_energy) * 100
    
    print(f"能量变化: {energy_reduction:+.2f}%")
    
    # 检测脉搏率
    freqs, spec = compute_spectrum(final_reconstructed, sampling_rate)
    pulse_range = (freqs >= 0.8) & (freqs <= 2.5)
    if np.any(pulse_range):
        pulse_freq = freqs[pulse_range][np.argmax(spec[pulse_range])]
        pulse_bpm = pulse_freq * 60
        print(f"检测到脉搏率: {pulse_bpm:.1f} BPM (频率: {pulse_freq:.2f} Hz)")
    
    print(f"\n处理策略总结:")
    for log in processing_log:
        print(f"  - {log}")
    
else:
    print("EEMD分解失败或没有有效的IMF分量")

print(f"\n处理完成！第{plot_column_index+1}列信号的处理完成。")