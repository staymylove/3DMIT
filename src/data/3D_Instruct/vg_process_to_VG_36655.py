import json
# with open('./src/data/3D_Instruct/meta_file/scanrefer_train_stage2_grounding_new.json','r') as file:
#     data_vg=json.load(file)     #36665   
# print('vg',len(data_vg))
# for vg in data_vg: 
#     path1='./src/data/3D_Instruct/scannet_pcls/'+vg['scene_id']+'_vert.npy'
#     vg['pcl']=path1
#     vg["task_type"]= "visual_grounding3d"
#     vg["src_dataset"]= "scanrefer"
#     vg["src_id"]= vg['scene_id']
#     lsx=[]
#     dr1={}
#     dr2={}
#     dr1["from"]="human"
#     dr2["from"]="gpt"
#     dr1["value"]=vg['prompt']
#     dr2["value"]=vg['ref_captions']
#     lsx.append(dr1)
#     lsx.append(dr2)
#     vg["conversations"]=lsx

# with open('./src/data/3D_Instruct/meta_file/VG_36655_chat3dv2.json','w') as file_3rscan:
#     json.dump(data_vg,file_3rscan,indent=4)

# print('done')



#val:

with open('./src/data/3D_Benchmark/meta_file/VG_ScanRefer_val_grounding_new.json','r') as file:
    data_vg=json.load(file)     #36665 
for vg in data_vg: 
    path1='./src/data/3D_Instruct/scannet_pcls/'+vg['scene_id']+'_vert.npy'
    vg['pcl']=path1
    vg["src_dataset"]= "scanrefer"
    vg["id"]= vg['scene_id']
    vg["question"]=vg['prompt']
    vg['query']=vg['prompt']
    vg['object']=vg['ref_captions']
    vg['task_type']='visual_grounding3d'
with open('./src/data/3D_Instruct/meta_file/VG_ScanRefer_val_grounding_new.json','w') as file_3rscan:
    json.dump(data_vg,file_3rscan,indent=4)

print('done')