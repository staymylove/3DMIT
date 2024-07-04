import json
import jsonlines
import re
import random
# data='./ckpt/zeju-1222-1913-bs-1-74kdata-addobj-img-1024-llava/answers-vg/zeju-12251033-bs-1-74kdata-addobj-1024-llava/VG_ScanRefer.jsonl'
data = './ckpt/zeju_0318_bs_1_onlyvg_vicuna_index/answers-vg/zeju-0318-bs-1-vg-vicuna-index/VG_ScanRefer.jsonl' 
locs_gt_file='./src/feature/scannet_attributes.json'
bench_vg_file='./src/data/3D_Benchmark/meta_file/VG_ScanRefer.json'

with open(bench_vg_file,'r') as file_bench:
    bench_vg=json.load(file_bench)
with open(locs_gt_file,'r') as file1:
    locs_vg=json.load(file1)

with open(data,'rb') as file:
    vg=jsonlines.Reader(file)
    pred_data=[]
    i=0
    for vg_data in vg:
        # print(vg_data)
        # print(vg_data['text'][0:2].isdigit())
        
        if 'Obj' not in vg_data['text'] and 'OBJ' not in vg_data['text'] and 'obj' not in vg_data['text']:
            if vg_data['text'][0:2].isdigit():
              
                vg_data['text']='OBJ'+vg_data['text'][0:2]
                
            elif not vg_data['text'][0:2].isdigit() and vg_data['text'][0:1].isdigit():
                vg_data['text']='OBJ0'+vg_data['text'][0:1] 
            else :
                vg_data['text']='OBJ'+str(random.randint(0, 30))+str(random.randint(0, 30))+'.\\n'
            
        if not vg_data['text'][3:5].isdigit():
            
            vg_data['text']='OBJ'+str(random.randint(0, 30))+str(random.randint(0, 30))+'.\\n'
            
        scene_id=vg_data['id']

        obj_id=vg_data['text'][3:5]
        if '.' in obj_id:
            obj_id = obj_id.replace('.', '')
        if '\n' in obj_id:
            obj_id = obj_id.replace('\n', '')
        
       
        obj_id=int(obj_id)
        # print(obj_id)
        # i=i+1
        # if i ==5:
        #     break
        if type(obj_id)!= int:
            print(obj_id)
        
        if obj_id < len(locs_vg[scene_id]['locs']):
            locs=locs_vg[scene_id]['locs'][obj_id]
        else:
            locs=locs_vg[scene_id]['locs'][0]
        pred_data.append(locs)
# print(pred_data)





ground_truth=[]
for gt in bench_vg:
    scene_id=gt['scene_id']
    obj_id = gt['obj_id']
    locs=locs_vg[scene_id]['locs'][obj_id]
    ground_truth.append(locs)

print(len(ground_truth))
# num_pattern =  re.compile(r'[0-9]+\.?[0-9]*')
num_pattern =  re.compile(r'[0-9]*')
def parse_entity(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in string.punctuation]
    words = [word for word in words if word not in stops]
    words = [wordnet.morphy(word) for word in words if word not in stops]
    return words

def is_number(s):
    try: 
        float(s)
        return True
    except ValueError: 
        pass 
    return False

def parse_num(num_list, split_char_a, split_char_b, text):
    flag = 0
    tmpnum = ''
    for c in text:
        if c == split_char_a:
            flag = 1
        elif c == split_char_b:
            flag = 0
            if is_number(tmpnum):
                num_list.append(float(tmpnum))
                tmpnum = ''
        elif flag == 0:
            continue
        else:
            if c!= ',' and c!= ' ':
                tmpnum += c
            else:
                if is_number(tmpnum):
                    num_list.append(float(tmpnum))
                    tmpnum = ''
    return num_list

def cal_iou_3d(bbox1, bbox2):
    '''
        box [x1, y1, z1, l, w, h]
    '''
    bbox1 = [
        round(bbox1[0] - abs(bbox1[3]/2), 3), round(bbox1[1] - abs(bbox1[4]/2), 3), round(bbox1[2] - abs(bbox1[5]/2), 3), 
        round(bbox1[0] + abs(bbox1[3]/2), 3), round(bbox1[1] + abs(bbox1[4]/2), 3), round(bbox1[2] + abs(bbox1[5])/2, 3)
        ]
    
    bbox2 = [
        round(bbox2[0]-abs(bbox2[3]/2),3), round(bbox2[1]-abs(bbox2[4]/2),3), round(bbox2[2]-abs(bbox2[5]/2),3), 
        round(bbox2[0]+abs(bbox2[3]/2),3), round(bbox2[1]+abs(bbox2[4]/2),3), round(bbox2[2]+abs(bbox2[5])/2,3)
        ]
    
    # intersection
    x1, y1, z1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), max(bbox1[2], bbox2[2])
    x2, y2, z2 = min(bbox1[3], bbox2[3]), min(bbox1[4], bbox2[4]), min(bbox1[5], bbox2[5])
    inter_area = (x2 - x1) * (y2 - y1) * (z2 - z1)

    # union
    area1 = (bbox1[3] - bbox1[0]) * (bbox1[4] - bbox1[1]) * (bbox1[5] - bbox1[2])
    area2 = (bbox2[3] - bbox2[0]) * (bbox2[4] - bbox2[1]) * (bbox2[5] - bbox2[2])
    uni_area = area1 + area2 - inter_area
    
    iou = inter_area / uni_area
    
    if iou > 1 or iou < 0:
        return 0
    else:
        return iou
    
def parse_bbox_3d(text):
    num_list = []
    num_list = parse_num(num_list, '[', ']', text)
    num_list = parse_num(num_list, '(', ')', text)
    print(num_list)
    bbox_list = []
    num_list = num_list[:(len(num_list) // 6) * 6]
    print(num_list)
    if len(num_list) == 0:
        # str_list = num_pattern.findall(text)
        # num_list = [float(item) for item in str_list]
        num_list = num_list[:(len(num_list)//6) * 6]
    for i in range(0,len(num_list), 6):
        cur_bbox = [num_list[j] for j in range(i, i + 6)]
        
        bbox_list.append(cur_bbox)
    return bbox_list




def grounding3d_eval(ground_truth, pred_data, thres=0.5):
    score = 0
    cnt = 0
    for i in range(len(ground_truth)):
        text = pred_data[i]  #[1,1,1,1,1,1]
        
        # bboxes = parse_bbox_3d(text)
       
        text_gt= ground_truth[i]
        # bboxes_gt = parse_bbox_3d(text_gt)
        cnt += 1
        # if len(bboxes) < 1:
        #     continue
        # bbox = bboxes[0]
        # bbox_gt = bboxes_gt[0]
        
        iou = cal_iou_3d(text, text_gt)
        
        if iou > thres:
            score += 1
    print("Acc over {}: {}".format(thres, score / cnt))


grounding3d_eval(ground_truth, pred_data, thres=0.25)
