import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_sensor_data(file_path):
    """加载传感器数据"""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        data_list = []
        for line in lines:
            values = line.strip().split(',')
            row = []
            for v in values:
                try:
                    row.append(float(v))
                except ValueError:
                    continue
            if len(row) >= 12:
                data_list.append(row[:12])
        
        sensor_data = np.array(data_list)
        
        print(f"数据加载成功：{sensor_data.shape[0]}行，{sensor_data.shape[1]}个传感器")
        print(f"数据范围：{np.min(sensor_data):.2f} - {np.max(sensor_data):.2f}")
        
        return sensor_data
        
    except Exception as e:
        print(f"数据加载错误: {e}")
        return generate_sample_data()

def generate_sample_data():
    """生成模拟数据用于测试"""
    n_points = 6000
    time = np.linspace(0, 12*np.pi, n_points)
    
    sensor_data = np.zeros((n_points, 12))
    for i in range(12):
        freq = 1.0 + i * 0.02
        phase = i * 0.5
        amplitude = 0.8 + 0.2 * np.sin(i * 0.3)
        
        base_wave = np.sin(freq * time + phase) * amplitude
        noise = np.random.normal(0, 0.05, n_points)
        sensor_data[:, i] = base_wave + noise + 1.0
    
    return sensor_data

def create_pulse_animation(sensor_data):
    """创建脉搏数据3D柱状图动画"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取数据范围
    z_min, z_max = np.min(sensor_data), np.max(sensor_data)
    
    # 定义3x4网格的坐标
    xpos, ypos = np.meshgrid(np.arange(4), np.arange(3))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(12)
    
    # 每个柱子的尺寸
    dx = dy = 0.8  # 柱子的宽度和深度
    dz = sensor_data[0]  # 初始高度
    
    # 创建初始柱状图
    bars = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='blue', alpha=0.8, shade=True)
    
    # 设置坐标轴
    ax.set_xlabel('传感器列 (0-3)')
    ax.set_ylabel('传感器行 (0-2)')
    ax.set_zlabel('压力值')
    ax.set_title('脉诊仪3D压力分布动画')
    ax.set_zlim(0, z_max * 1.1)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 2.5)
    
    def update(frame):
        """更新动画帧"""
        # 清除之前的柱子
        ax.clear()
        
        # 获取当前帧数据
        dz = sensor_data[frame]
        
        # 重新绘制柱子
        bars = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='blue', alpha=0.8, shade=True)
        
        # 重新设置坐标轴属性
        ax.set_xlabel('传感器列 (0-3)')
        ax.set_ylabel('传感器行 (0-2)')
        ax.set_zlabel('压力值')
        ax.set_zlim(0, z_max * 1.1)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 2.5)
        
        # 更新标题
        current_time = frame / 100
        ax.set_title(f'脉诊仪3D压力分布 - 时间: {current_time:.2f}秒')
        
        return bars,
    
    # 创建动画 - 使用更少的帧数提高性能
    total_frames = len(sensor_data)
    frames_to_use = range(0, total_frames, 10)  # 每10帧取一帧
    
    animation = FuncAnimation(fig, update, frames=frames_to_use, 
                            interval=100, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return animation

# 替代方案：使用散点图显示12个独立传感器
def create_scatter_animation(sensor_data):
    """创建散点图动画，显示12个独立传感器"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取数据范围
    z_min, z_max = np.min(sensor_data), np.max(sensor_data)
    
    # 定义3x4网格的坐标
    xpos, ypos = np.meshgrid(np.arange(4), np.arange(3))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    
    # 创建初始散点图
    scatter = ax.scatter(xpos, ypos, sensor_data[0], 
                        c=sensor_data[0], cmap='viridis', s=100)
    
    # 设置坐标轴
    ax.set_xlabel('传感器列 (0-3)')
    ax.set_ylabel('传感器行 (0-2)')
    ax.set_zlabel('压力值')
    ax.set_title('脉诊仪3D压力分布动画')
    ax.set_zlim(0, z_max * 1.1)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 2.5)
    
    # 添加颜色条
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, label='压力值')
    
    def update(frame):
        """更新动画帧"""
        # 清除之前的散点
        ax.clear()
        
        # 获取当前帧数据
        z = sensor_data[frame]
        
        # 重新绘制散点
        scatter = ax.scatter(xpos, ypos, z, c=z, cmap='viridis', s=100)
        
        # 重新设置坐标轴属性
        ax.set_xlabel('传感器列 (0-3)')
        ax.set_ylabel('传感器行 (0-2)')
        ax.set_zlabel('压力值')
        ax.set_zlim(0, z_max * 1.1)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_title(f'脉诊仪3D压力分布 - 时间: {frame/100:.2f}秒')
        
        return scatter,
    
    # 创建动画
    total_frames = len(sensor_data)
    frames_to_use = range(0, total_frames, 10)
    
    animation = FuncAnimation(fig, update, frames=frames_to_use, 
                            interval=100, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return animation

# 主程序
if __name__ == "__main__":
    # 加载数据
    file_path = r"D:\python_project\data\cleaned_data_2025-09-19_10-16-28.csv"
    sensor_data = load_sensor_data(file_path)
    
    # 让用户选择可视化方式
    print("请选择可视化方式:")
    print("1. 3D柱状图 (显示12个独立传感器)")
    print("2. 3D散点图 (显示12个独立传感器)")
    
    choice = input("请输入选择 (1或2): ")
    
    try:
        if choice == "1":
            print("创建3D柱状图动画中...")
            animation = create_pulse_animation(sensor_data)
        else:
            print("创建3D散点图动画中...")
            animation = create_scatter_animation(sensor_data)
    except Exception as e:
        print(f"动画创建失败: {e}")
        print("尝试使用静态图展示...")
        
        # 静态展示几个关键帧
        fig = plt.figure(figsize=(15, 10))
        time_points = np.linspace(0, len(sensor_data)-1, 6, dtype=int)
        
        for i, frame in enumerate(time_points):
            ax = fig.add_subplot(2, 3, i+1, projection='3d')
            
            xpos, ypos = np.meshgrid(np.arange(4), np.arange(3))
            xpos = xpos.flatten()
            ypos = ypos.flatten()
            z = sensor_data[frame]
            
            scatter = ax.scatter(xpos, ypos, z, c=z, cmap='viridis', s=100)
            ax.set_title(f'时间: {frame/100:.2f}秒')
            ax.set_xlabel('列')
            ax.set_ylabel('行')
            ax.set_zlabel('压力')
            
            fig.colorbar(scatter, ax=ax, shrink=0.6)
        
        plt.tight_layout()
        plt.show()