import pandas as pd
import re

def clean_and_read_data(file_path):
    """
    读取并清理文本文件，提取数值数据，兼容逗号和空格分隔
    
    参数:
    file_path: 文本文件的路径
    
    返回:
    包含数据的DataFrame
    """
    # 读取原始文件
    with open(file_path, 'r', encoding='gbk') as file:
        raw_lines = file.readlines()
    
    print(f"原始文件行数: {len(raw_lines)}")
    
    cleaned_lines = []
    for i, line in enumerate(raw_lines):
        # 打印前几行原始数据用于调试
        if i < 5:
            print(f"原始行 {i}: {repr(line.strip())}")  # 使用repr显示原始内容
        
        # 移除行首行尾空白字符
        cleaned_line = line.strip()
        
        # 跳过空行
        if not cleaned_line:
            continue
            
        # 尝试用逗号分割
        if ',' in cleaned_line:
            parts = cleaned_line.split(',')
        # 如果没有逗号，尝试用空格分割
        else:
            # 使用split()而不是split(' ')可以处理多个连续空格
            parts = cleaned_line.split()
        
        # 过滤空字符串并确保有足够的数据
        parts = [p.strip() for p in parts if p.strip()]
        
        if i < 5:
            print(f"分割后行 {i}: {parts}")
        
        if len(parts) >= 14:  # 确保至少有14列
            cleaned_lines.append(parts[:14])  # 只取前14列
        else:
            print(f"警告: 第{i}行只有{len(parts)}列，跳过")
    
    print(f"有效数据行数: {len(cleaned_lines)}")
    
    if not cleaned_lines:
        print("错误: 没有找到有效数据")
        return pd.DataFrame()
    
    # 将清理后的数据转换为DataFrame
    df = pd.DataFrame(cleaned_lines)
    print(f"创建的DataFrame形状: {df.shape}")
    
    # 转换为数值类型
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# 使用示例
file_path = r'D:\python_project\data\cleaned_data_2025-10-01_11-07-26.csv'
data_df = clean_and_read_data(file_path)

# 只有在DataFrame不为空且列数正确时才设置列名
if not data_df.empty:
    if data_df.shape[1] == 14:
        data_df.columns = ['channel_1', 'channel_2', 'channel_3', 'channel_4','channel_5' , 'channel_6','channel_7',
                           'channel_8','channel_9','channel_10','channel_11','channel_12','pressure','electric']
        
        print("前几行数据：")
        print(data_df.head())
        
        # 保存数据
        #save_path = r'D:\python_project\data\cleaned_data_2025_10_27_17_51_01.csv'
        #data_df.to_csv(save_path, sep=',', index=False, header=False, encoding='utf-8')
        #print(f"数据已保存到: {save_path}")
        
        # 绘图代码保持不变
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 绘制前12列数据
        if data_df.shape[1] >= 12:
            fig, axs = plt.subplots(4, 3, figsize=(15, 9))
            axs = axs.flatten()
            
            for i in range(12):
                column_name = data_df.columns[i]
                y_data = data_df[column_name]
                x_data = range(len(y_data))
                
                axs[i].plot(x_data, y_data, 'b-', label=column_name)
                axs[i].set_title(f'Waveform: {column_name}')
                axs[i].set_xlabel('')
                axs[i].set_ylabel('Amplitude')
                axs[i].legend()
                axs[i].grid(False)
                
                y_min = y_data.min()
                y_max = y_data.max()
                yticks = np.arange(int(y_min) - 20, int(y_max) + 20, 100)
                axs[i].set_yticks(yticks)
            
            plt.tight_layout()
            plt.show()
        
        # 绘制第13列数据
        if data_df.shape[1] >= 13:
            plt.figure(figsize=(8, 6))
            column_index_13 = data_df.columns[12]
            y_data_13 = data_df[column_index_13]
            x_data_13 = range(len(y_data_13))
            
            plt.plot(x_data_13, y_data_13, 'b-', label=column_index_13)
            plt.title(f'Waveform of {column_index_13} Data') 
            plt.xlabel('Sample Index') 
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(False)
            
            y_min_13 = y_data_13.min()
            y_max_13 = y_data_13.max()
            yticks_13 = np.arange(int(y_min_13) - 20, int(y_max_13) + 20, 50)
            plt.yticks(yticks_13)
            
            plt.show()
    else:
        print(f"数据列数不正确，期望14列，实际{data_df.shape[1]}列")
        print("前几行数据：")
        print(data_df.head())
else:
    print("DataFrame为空")