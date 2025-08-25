import pickle
import os

# --- 配置 ---
# 数据根目录
root_path = '/home/lyh/dataspace/lorel/may_08_sawyer_50k'
# 输入的 pkl 文件名
input_filename = 'prep_data.pkl'
# 输出的 pkl 文件名
output_filename = 'prep_data_first_200.pkl'
# 要提取的数据条数
num_items_to_keep = 200

# --- 主逻辑 ---
input_filepath = os.path.join(root_path, input_filename)
output_filepath = os.path.join(root_path, output_filename)

print(f"正在读取文件: {input_filepath}")

# 以二进制读取模式打开 pkl 文件
with open(input_filepath, 'rb') as f_in:
    # 加载数据，通常是一个字典，其值为Numpy数组
    original_data = pickle.load(f_in)

# 创建一个新的字典来存放切片后的数据
sliced_data = {}

print("正在处理数据，提取前200条...")

# 遍历原始数据字典中的每一项
for key, value in original_data.items():
    print(f"  - 正在切片键: '{key}'，原始形状: {value.shape}")
    # 对每个值（通常是Numpy数组）进行切片，只取第一个维度的前200条
    sliced_data[key] = value[:num_items_to_keep]
    print(f"    新形状: {sliced_data[key].shape}")

# 以二进制写入模式打开新的 pkl 文件
with open(output_filepath, 'wb') as f_out:
    # 将切片后的数据字典写入新文件
    pickle.dump(sliced_data, f_out)

print(f"\n处理完成！已将前 {num_items_to_keep} 条数据保存到: {output_filepath}")