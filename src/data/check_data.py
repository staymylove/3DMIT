import numpy as np
import os  
  
def read_npy_files(directory):  
    file_count = 0
    right_file = 0
    for filename in os.listdir(directory):  
        if filename.endswith('.npy'):
            file_count += 1
            filepath = os.path.join(directory, filename)
            try:  
                data = np.load(filepath)  
                right_file += 1
                # print(f"Loaded {filename} with shape {data.shape}")
            except Exception as e:  
                print(f"Error reading file {filepath}: {str(e)}")  
            
    print(f'load files: {right_file} / {file_count}')
  
# 用你希望遍历的文件夹替换 'your_directory'  
read_npy_files('./src/data/3D_Instruct/3rscan_pcls')

# _3rscan_data_path = './src/data/3D_Instruct/3rscan_pcls/0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca.npy'
# shapenet_data_path = './src/data/3D_Instruct/shapenet_pcls/02691156_1a6ad7a24bb89733f412783097373bdc_4096.npy'
# scanrefer_data_path = './src/data/3D_Benchmark/scannet_pcls/scene0011_00_vert.npy'
# _3rscan_data = np.load(_3rscan_data_path)  # [num_point, 3]
# shapenet_data = np.load(shapenet_data_path)  # [num_point, 6(xyz+rgb)]
# scanrefer_data = np.load(scanrefer_data_path)  # [50000, 6(xyz+rgb)]

pass