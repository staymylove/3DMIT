import json

def gen_scanqa(split = 'val'):
    if split == 'train':
        output_path = './src/data/scanqa_v1.0/ScanQA_v1.0_train_lamm.json'
        with open('./src/data/scanqa_v1.0/ScanQA_v1.0_train.json', 'r') as f:
            data = json.load(f)
        res = []
        # update gpt values 
        for item in data:
            pcl = './src/data/3D_Instruct/scannet_pcls/'+item['scene_id']+'_vert.npy'
            conversations = [
                {
                    "from": "human",
                    "value": item['question']
                },
                {
                    "from": "gpt",
                    "value": item['answers'][0]
                }
            ]
            res.append({
                "id": item['scene_id'],
                "pcl": pcl,
                "conversations": conversations,
                'task_type' : "VQA3D",
                'src_dataset' : "scanqa"
            })

        # 将键值对保存到JSON文件中  
        with open(output_path, 'w') as file:
            json.dump(res, file, indent=4)
    else:
        output_path = './src/data/scanqa_v1.0/ScanQA_v1.0_val_3dmit.json'
        with open('./src/data/scanqa_v1.0/ScanQA_v1.0_val.json', 'r') as f:
            data = json.load(f)
        res = []
        # update gpt values 
        for item in data:
            pcl = './src/data/3D_Instruct/scannet_pcls/'+item['scene_id']+'_vert.npy'
            res.append({
                'question_id': item['question_id'],
                "pcl": pcl,
                "id": item['scene_id'],
                "query": item['question'],
                'task_type' : "VQA3D",
                'src_dataset' : "scanqa",
                'gt_choices': item['answers']
            })

        # 将键值对保存到JSON文件中  
        with open(output_path, 'w') as file:
            json.dump(res, file, indent=4)
    pass


if __name__ == '__main__':
    gen_scanqa(split='val')