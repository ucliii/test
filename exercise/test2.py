import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import warnings

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

# 参数设置
fs = 250  # 采样率
T = 20    # 信号时长 (秒)
t = np.linspace(0, T, int(fs*T), endpoint=False)

# ==================== 在这里修改噪声频率 ====================
noise_freq = 0.5  # 基线漂移噪声的主频 (Hz)
# =========================================================

# 1. 创建主频为3Hz的规则脉搏波信号
f_main = 3.0  # 信号主频 (Hz)

# 创建脉搏波信号
clean_signal = (1.5 * np.sin(2*np.pi*f_main*t) + 
                0.8 * np.sin(2*np.pi*2*f_main*t) + 
                0.5 * np.sin(2*np.pi*3*f_main*t))

# 归一化
clean_signal = clean_signal / np.max(np.abs(clean_signal)) * 2.0

# 2. 创建基线漂移噪声
baseline_noise = 0.3 * np.sin(2*np.pi*noise_freq*t)

# 3. 创建肌肉抽搐噪声 - 既有向上也有向下的脉冲
np.random.seed(100)
num_pulses = 20
pulse_positions = np.random.randint(0, len(t), num_pulses)
pulse_amplitudes = np.random.uniform(0.5, 1.0, num_pulses)
pulse_directions = np.random.choice([-1, 1], num_pulses)  # 随机选择脉冲方向

muscle_noise = np.zeros_like(t)
for pos, amp, direction in zip(pulse_positions, pulse_amplitudes, pulse_directions):
    pulse_width = int(0.1 * fs)
    start = max(0, pos - pulse_width//2)
    end = min(len(t), pos + pulse_width//2)
    
    pulse_t = np.linspace(-2, 2, end - start)
    pulse = amp * np.exp(-pulse_t**2 / 0.5) * direction  # 乘以方向
    muscle_noise[start:end] += pulse

# 4. 创建混合噪声和混合信号
mixed_noise = 2*baseline_noise + 4*muscle_noise
mixed_signal = clean_signal + mixed_noise

# 5. 设计巴特沃斯带通滤波器
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# 设置滤波器参数
lowcut = 0.8
highcut = 10
order = 2

# 应用滤波器
filtered_signal = butter_bandpass_filter(mixed_signal, lowcut, highcut, fs, order)

# 6. 计算频谱
def compute_spectrum(signal, fs):
    n = len(signal)
    freq = fftfreq(n, 1/fs)
    spectrum = np.abs(fft(signal)) / n
    return freq[:n//2], 2 * spectrum[:n//2]

freq_clean, spec_clean = compute_spectrum(clean_signal, fs)
freq_mixed_noise, spec_mixed_noise = compute_spectrum(mixed_noise, fs)
freq_mixed, spec_mixed = compute_spectrum(mixed_signal, fs)
freq_filtered, spec_filtered = compute_spectrum(filtered_signal, fs)

# 7. 绘制8幅图
plt.figure(figsize=(16, 12))

# 第1行：原始信号
plt.subplot(4, 2, 1)
plt.plot(t, clean_signal, 'b', linewidth=1.5)
plt.title('原始信号 (时域)')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)

plt.subplot(4, 2, 2)
plt.plot(freq_clean, spec_clean, 'b', linewidth=2)
plt.axvline(x=f_main, color='blue', linestyle='--', alpha=0.7, label=f'主频: {f_main}Hz')
plt.title('原始信号 (频域)')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅度')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)

# 第2行：混合噪声
plt.subplot(4, 2, 3)
plt.plot(t, mixed_noise, 'r', linewidth=1.5)
plt.title('混合噪声 (时域)')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)

plt.subplot(4, 2, 4)
plt.plot(freq_mixed_noise, spec_mixed_noise, 'r', linewidth=2)
plt.axvline(x=noise_freq, color='red', linestyle='--', alpha=0.7, label=f'基线: {noise_freq}Hz')
plt.title('混合噪声 (频域)')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅度')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)

# 第3行：混合信号
plt.subplot(4, 2, 5)
plt.plot(t, mixed_signal, 'g', linewidth=1.5)
plt.title('混合信号 (时域)')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)

plt.subplot(4, 2, 6)
plt.plot(freq_mixed, spec_mixed, 'g', linewidth=2)
plt.axvline(x=f_main, color='blue', linestyle='--', alpha=0.7, label=f'信号: {f_main}Hz')
plt.axvline(x=noise_freq, color='red', linestyle='--', alpha=0.7, label=f'噪声: {noise_freq}Hz')
plt.title('混合信号 (频域)')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅度')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)

# 第4行：滤波后信号
plt.subplot(4, 2, 7)
plt.plot(t, filtered_signal, 'purple', linewidth=1.5)
plt.title('滤波后信号 (时域)')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)

plt.subplot(4, 2, 8)
plt.plot(freq_filtered, spec_filtered, 'purple', linewidth=2)
plt.axvline(x=f_main, color='blue', linestyle='--', alpha=0.7, label=f'信号: {f_main}Hz')
plt.axvspan(lowcut, highcut, alpha=0.2, color='gray', label='滤波器通带')
plt.title('滤波后信号 (频域)')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅度')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)

plt.tight_layout()
plt.show()

print(f"当前设置:")
print(f"  信号主频: {f_main}Hz")
print(f"  基线漂移噪声主频: {noise_freq}Hz")
print(f"  滤波器参数: {lowcut}-{highcut}Hz, 阶数{order}")
print(f"  采样率: {fs}Hz")
print(f"  原始信号幅度范围: [{np.min(clean_signal):.3f}, {np.max(clean_signal):.3f}]")
print(f"  混合信号幅度范围: [{np.min(mixed_signal):.3f}, {np.max(mixed_signal):.3f}]")
print(f"  滤波后信号幅度范围: [{np.min(filtered_signal):.3f}, {np.max(filtered_signal):.3f}]")