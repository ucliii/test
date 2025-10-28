import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.fft import fft, fftfreq, ifft
import warnings

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

# 设置采样参数
fs = 1000  # 提高采样率以捕捉快速变化
t = np.linspace(0, 4, 4*fs, endpoint=False)

# 创建真正的非平稳信号 - 包含多种时变特征
signal = np.zeros_like(t)

# 1. 频率线性变化的成分 (从5Hz到25Hz)
chirp_freq = 5 + 20 * t / 4  # 线性变化
signal += 0.7 * np.sin(2 * np.pi * chirp_freq * t)

# 2. 瞬时脉冲 - 在1.5秒和3秒处
pulse1 = np.exp(-100 * (t - 1.5)**2) * 3  # 高斯脉冲
pulse2 = np.exp(-80 * (t - 3.0)**2) * 2   # 另一个高斯脉冲
signal += pulse1 + pulse2

# 3. 频率跳变 - 在2秒处从15Hz跳变到8Hz
freq_jump = np.zeros_like(t)
freq_jump[t < 2] = np.sin(2 * np.pi * 15 * t[t < 2])
freq_jump[t >= 2] = np.sin(2 * np.pi * 8 * t[t >= 2])
signal += 0.5 * freq_jump

# 4. 添加一些噪声
np.random.seed(42)
noise = 0.1 * np.random.randn(len(t))
signal += noise

# 计算傅里叶变换
fft_signal = fft(signal)
freqs = fftfreq(len(t), 1/fs)

# 第一张图：原始信号和频谱
plt.figure(figsize=(15, 10))

# 原始信号
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('复杂非平稳信号: 线性调频 + 瞬时脉冲 + 频率跳变 + 噪声')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.grid(True)

# 频谱
plt.subplot(2, 1, 2)
positive_freq_idx = freqs >= 0
freqs_positive = freqs[positive_freq_idx]
fft_positive = fft_signal[positive_freq_idx]
plt.plot(freqs_positive, np.abs(fft_positive))
plt.title('信号频谱 - 无法识别时变特征')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅度')
plt.xlim(0, 50)
plt.grid(True)

plt.tight_layout()
plt.show()

# 第二张图：去噪效果对比
plt.figure(figsize=(15, 12))

# 原始信号
plt.subplot(4, 1, 1)
plt.plot(t, signal)
plt.title('原始信号')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.grid(True)

# 傅里叶变换去噪 - 去除高频噪声
fft_filtered = fft_signal.copy()
# 尝试去除高频噪声 (30Hz以上)
freq_mask_high = np.abs(freqs) > 30
fft_filtered[freq_mask_high] = 0
signal_fourier_filtered = np.real(ifft(fft_filtered))

plt.subplot(4, 1, 2)
plt.plot(t, signal_fourier_filtered)
plt.title('傅里叶变换去噪 (去除30Hz以上成分)')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.grid(True)

# 小波变换去噪 - 使用自适应阈值
wavelet = 'db8'
level = 6
coeffs = pywt.wavedec(signal, wavelet, level=level)

# 计算每个细节系数的噪声标准差估计
sigma = np.median(np.abs(coeffs[-1])) / 0.6745

# 对每个细节系数应用软阈值
coeffs_thresh = []
for i, coeff in enumerate(coeffs):
    if i == 0:  # 近似系数不处理
        coeffs_thresh.append(coeff)
    else:
        # 使用通用阈值，但可以根据需要调整
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs_thresh.append(pywt.threshold(coeff, threshold, mode='soft'))

signal_wavelet_filtered = pywt.waverec(coeffs_thresh, wavelet)

if len(signal_wavelet_filtered) > len(t):
    signal_wavelet_filtered = signal_wavelet_filtered[:len(t)]

plt.subplot(4, 1, 3)
plt.plot(t, signal_wavelet_filtered)
plt.title('小波变换去噪 (自适应软阈值)')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.grid(True)

# 对比两种方法
plt.subplot(4, 1, 4)
plt.plot(t, signal_fourier_filtered, 'r-', label='傅里叶滤波', alpha=0.8)
plt.plot(t, signal_wavelet_filtered, 'g--', label='小波滤波', alpha=0.8)
plt.title('两种方法对比')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 第三张图：小波变换的真正优势 - 时频分析
plt.figure(figsize=(15, 10))

# 小波时频谱 - 展示信号的时变特性
scales = np.arange(1, 128)
coefficients, frequencies = pywt.cwt(signal, scales, 'morl', sampling_period=1/fs)

plt.subplot(2, 1, 1)
plt.imshow(np.abs(coefficients), extent=[0, 4, 1, 50], aspect='auto', 
           cmap='jet', origin='lower')
plt.colorbar(label='小波系数幅度')
plt.title('小波时频谱 - 清晰展示信号的时变特征')
plt.ylabel('尺度 (对应频率)')
plt.xlabel('时间 (s)')

# 对比：短时傅里叶变换
from scipy.signal import stft
f_stft, t_stft, Zxx = stft(signal, fs=fs, nperseg=256)

plt.subplot(2, 1, 2)
plt.pcolormesh(t_stft, f_stft, np.abs(Zxx), shading='gouraud', cmap='jet')
plt.colorbar(label='STFT幅度')
plt.title('短时傅里叶变换 - 时频分辨率受限')
plt.ylabel('频率 (Hz)')
plt.xlabel('时间 (s)')
plt.ylim(0, 50)

plt.tight_layout()
plt.show()

# 定量评估 - 重点关注脉冲和频率跳变的保留情况
print("=" * 70)
print("非平稳信号处理能力评估:")
print("=" * 70)

# 计算关键特征的保留程度
def measure_feature_preservation(original, filtered, feature_times, window=0.1):
    """测量特定时间点特征的保留程度"""
    preservation_ratios = []
    for feature_time in feature_times:
        # 提取特征窗口
        idx = (t >= feature_time - window/2) & (t <= feature_time + window/2)
        orig_energy = np.sum(original[idx]**2)
        filt_energy = np.sum(filtered[idx]**2)
        preservation = filt_energy / orig_energy if orig_energy > 0 else 0
        preservation_ratios.append(preservation)
    return np.mean(preservation_ratios)

# 关键特征时间点
pulse_times = [1.5, 3.0]  # 两个脉冲
jump_time = [2.0]  # 频率跳变点

# 计算特征保留度
fourier_pulse_preservation = measure_feature_preservation(signal, signal_fourier_filtered, pulse_times)
wavelet_pulse_preservation = measure_feature_preservation(signal, signal_wavelet_filtered, pulse_times)

fourier_jump_preservation = measure_feature_preservation(signal, signal_fourier_filtered, jump_time)
wavelet_jump_preservation = measure_feature_preservation(signal, signal_wavelet_filtered, jump_time)

# 计算整体噪声减少程度
original_noise_level = np.std(noise)
fourier_noise_reduction = np.std(signal_fourier_filtered - (signal - noise))
wavelet_noise_reduction = np.std(signal_wavelet_filtered - (signal - noise))

print(f"\n脉冲特征保留度:")
print(f"  傅里叶变换: {fourier_pulse_preservation*100:.1f}%")
print(f"  小波变换: {wavelet_pulse_preservation*100:.1f}%")

print(f"\n频率跳变特征保留度:")
print(f"  傅里叶变换: {fourier_jump_preservation*100:.1f}%")
print(f"  小波变换: {wavelet_jump_preservation*100:.1f}%")

print(f"\n噪声抑制效果 (越小越好):")
print(f"  傅里叶变换残留噪声: {fourier_noise_reduction:.4f}")
print(f"  小波变换残留噪声: {wavelet_noise_reduction:.4f}")

print("\n" + "=" * 70)
print("关键结论:")
print("1. 傅里叶变换: 对非平稳信号处理能力有限，会模糊时变特征")
print("2. 小波变换: 能更好地保留瞬时脉冲和频率跳变等时变特征")
print("3. 小波时频谱: 清晰展示信号的时频结构，优于短时傅里叶变换")
print("4. 对于真正的非平稳信号，小波变换展现出明显优势")