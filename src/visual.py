import numpy as np
import json
from plyfile import (PlyData, PlyElement)
import argparse
import os
'''
inputs: 
    - 场景点云信息_vh_clean_2.ply
    - 对齐矩阵.txt
    - bbox信息 (center + size) or (8 corners)

outputs:
    - 包含bbox可视化的点云文件
'''

def json_read(filename):
	with open(filename, 'r') as infile:
		return json.load(infile)


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])


def get_3d_box(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
    '''
    R = roty(heading_angle)
    l,w,h = box_size

    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]

    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))

    corners_3d[0,:] = corners_3d[0,:] + center[0]
    corners_3d[1,:] = corners_3d[1,:] + center[1]
    corners_3d[2,:] = corners_3d[2,:] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def get_bbox_ply(bbox_info, mode, bbox_mode = 'bbox'):
    """
    bbox: (cx, cy, cz, lx, ly, lz, r), center and length in three axis, the last is the rotation
    output_file: string

    """
    def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):

        import math

        def compute_length_vec3(vec3):
            return math.sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2])

        def rotation(axis, angle):
            rot = np.eye(4)
            c = np.cos(-angle)
            s = np.sin(-angle)
            t = 1.0 - c
            axis /= compute_length_vec3(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot[0,0] = 1 + t*(x*x-1)
            rot[0,1] = z*s+t*x*y
            rot[0,2] = -y*s+t*x*z
            rot[1,0] = -z*s+t*x*y
            rot[1,1] = 1+t*(y*y-1)
            rot[1,2] = x*s+t*y*z
            rot[2,0] = y*s+t*x*z
            rot[2,1] = -x*s+t*y*z
            rot[2,2] = 1+t*(z*z-1)
            return rot


        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        for i in range(stacks+1):
            for i2 in range(slices):
                theta = i2 * 2.0 * math.pi / slices
                pos = np.array([radius*math.cos(theta), radius*math.sin(theta), height*i/stacks])
                verts.append(pos)
        for i in range(stacks):
            for i2 in range(slices):
                i2p1 = math.fmod(i2 + 1, slices)
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2, i*slices + i2p1], dtype=np.uint32) )
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2p1, (i + 1)*slices + i2p1], dtype=np.uint32) )
        transform = np.eye(4)
        va = np.array([0, 0, 1], dtype=np.float32)
        vb = diff
        vb /= compute_length_vec3(vb)
        axis = np.cross(vb, va)
        angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        if angle != 0:
            if compute_length_vec3(axis) == 0:
                dotx = va[0]
                if (math.fabs(dotx) != 1.0):
                    axis = np.array([1,0,0]) - dotx * va
                else:
                    axis = np.array([0,1,0]) - va[1] * va
                axis /= compute_length_vec3(axis)
            transform = rotation(axis, -angle)
        transform[:3,3] += p0
        verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
        verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]
            
        return verts, indices

    def get_bbox_edges(bbox_min, bbox_max):
        def get_bbox_verts(bbox_min, bbox_max):
            verts = [
                np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),

                np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_max[2]])
            ]
            return verts

        box_verts = get_bbox_verts(bbox_min, bbox_max)
        edges = [
            (box_verts[0], box_verts[1]),
            (box_verts[1], box_verts[2]),
            (box_verts[2], box_verts[3]),
            (box_verts[3], box_verts[0]),

            (box_verts[4], box_verts[5]),
            (box_verts[5], box_verts[6]),
            (box_verts[6], box_verts[7]),
            (box_verts[7], box_verts[4]),

            (box_verts[0], box_verts[4]),
            (box_verts[1], box_verts[5]),
            (box_verts[2], box_verts[6]),
            (box_verts[3], box_verts[7])
        ]
        return edges

    def get_bbox_corners(bbox):
        centers, lengths = bbox[:3], bbox[3:6]
        xmin, xmax = centers[0] - lengths[0] / 2, centers[0] + lengths[0] / 2
        ymin, ymax = centers[1] - lengths[1] / 2, centers[1] + lengths[1] / 2
        zmin, zmax = centers[2] - lengths[2] / 2, centers[2] + lengths[2] / 2
        corners = []
        corners.append(np.array([xmax, ymax, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymax, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymax, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymax, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymin, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymin, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymin, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymin, zmax]).reshape(1, 3))
        corners = np.concatenate(corners, axis=0) # 8 x 3

        return corners

    radius = 0.03
    offset = [0,0,0]
    verts = []
    indices = []
    colors = []
    corners = get_bbox_corners(bbox_info) if bbox_mode == 'bbox' else np.concatenate([np.array(corner).reshape(1, 3) for corner in bbox_info], axis=0)

    box_min = np.min(corners, axis=0)
    box_max = np.max(corners, axis=0)
    palette = {
        0: [0, 255, 0], # gt
        1: [0, 0, 255]  # pred
    }
    chosen_color = palette[mode]
    edges = get_bbox_edges(box_min, box_max)
    for k in range(len(edges)):
        cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
        cur_num_verts = len(verts)
        cyl_color = [[c / 255 for c in chosen_color] for _ in cyl_verts]
        cyl_verts = [x + offset for x in cyl_verts]
        cyl_ind = [x + cur_num_verts for x in cyl_ind]
        verts.extend(cyl_verts)
        indices.extend(cyl_ind)
        colors.extend(cyl_color)
    
    return verts, colors, indices


def vis_bbox(scene_ply_path,
                scene_txt_path,
                bbox_info_path,
                mode,
                output
                ):
    '''
    bbox_info_path: 执行eval.py命令(python scripts/eval.py --qa --force), 生成pred_val.json, 包括了gt_box和pred_box信息
    '''
    
    # 计算给定场景的变换对齐矩阵
    lines = open(scene_txt_path).readlines()
    axis_align_matrix = None
    for line in lines:
         if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    Mscan = np.array(axis_align_matrix).reshape((4,4))

    # 读取bbox信息
    with open(bbox_info_path, "r") as json_file:
        bbox_info = json.load(json_file)

    aligned_mesh_faces = []
    aligned_mesh_vertices = []

    # 读取场景点云，并对齐坐标轴
    with open(scene_ply_path, 'rb') as read_file:
        mesh_scan = PlyData.read(read_file)
    for f in mesh_scan["face"]: 
        aligned_mesh_faces.append((np.array(f[0]),))
    for v in mesh_scan["vertex"]: 
        v1 = np.array([v[0], v[1], v[2], 1])
        v1 = np.dot(Mscan, v1)
        aligned_mesh_vertices.append(tuple(v1[:3]) + (v[3], v[4], v[5]))

    k = 0
    # 读取给定场景的bbox信息，并构造矩形框进行显示
    for align_info in bbox_info:
        if k >= 4:
            break
        if align_info["scene_id"] == os.path.basename(scene_ply_path)[:12]:
            if align_info["iou"] >= 0.7:
                k += 1
                if 'pred' in mode:
                    # pred bbox
                    pred_bbox_verts, pred_bbox_colors, pred_bbox_indices = get_bbox_ply(bbox_info=align_info['bbox'],
                                                                        mode=1,
                                                                        bbox_mode='corner')
                    for ind in pred_bbox_indices:
                        aligned_mesh_faces.append((np.array([
                            ind[0] + len(aligned_mesh_vertices), ind[1] + len(aligned_mesh_vertices), ind[2] + len(aligned_mesh_vertices)]),))
                    for bbox_vert, bbox_color in zip(pred_bbox_verts, pred_bbox_colors):
                        aligned_mesh_vertices.append((bbox_vert[0], bbox_vert[1], bbox_vert[2], int(bbox_color[0]*255), int(bbox_color[1]*255), int(bbox_color[2]*255)))
                
                if 'gt' in mode:
                    # gt bbox
                    gt_bbox_verts, gt_bbox_colors, gt_bbox_indices = get_bbox_ply(bbox_info=align_info['gt_bbox'],
                                                                        mode=0,
                                                                        bbox_mode='corner')
                    for ind in gt_bbox_indices:
                        aligned_mesh_faces.append((np.array([
                            ind[0] + len(aligned_mesh_vertices), ind[1] + len(aligned_mesh_vertices), ind[2] + len(aligned_mesh_vertices)]),))
                    for bbox_vert, bbox_color in zip(gt_bbox_verts, gt_bbox_colors):
                        aligned_mesh_vertices.append((bbox_vert[0], bbox_vert[1], bbox_vert[2], int(bbox_color[0]*255), int(bbox_color[1]*255), int(bbox_color[2]*255)))

    # 赋予点云vertices和faces属性
    aligned_mesh_vertices = np.asarray(aligned_mesh_vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    aligned_mesh_faces = np.asarray(aligned_mesh_faces, dtype=[('vertex_indices', 'i4', (3,))])

    # 写入点云文件
    objdata = PlyData([PlyElement.describe(aligned_mesh_vertices, 'vertex', comments=['vertices']),  
                       PlyElement.describe(aligned_mesh_faces, 'face', comments=['faces'])], comments=['scannet_with_bbox'])

    
    os.makedirs(os.path.dirname(output), exist_ok=True)
    output = output[:-4] + f'_num{str(k)}' + '.ply'
    with open(output, mode='wb') as f:
        PlyData(objdata).write(f)
    print("Saved Successfully in ", output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualization')
    parser.add_argument("--scene_ply", type=str, default='/data/renruilong/datasets/scannet/scans/scene0389_00/scene0389_00_vh_clean_2.ply', help='the path of xxx_vh_clean_2.ply')
    parser.add_argument("--scene_txt", type=str, default='/data/renruilong/datasets/scannet/scans/scene0389_00/scene0389_00.txt', help='the path of xxx.txt, witch includes alignment matrix')
    parser.add_argument("--bbox_info", type=str, default='/data/renruilong/code/scanqa/outputs/2023-08-24_15-40-33_XYZ_COLOR/pred.val.json', help='the path of bbox info with 8 corners or center_size format')
    parser.add_argument("--mode", type=str, default='pred_gt', help='the way to visualize bbox', choices=['pred_gt', 'pred', 'gt'])
    parser.add_argument("--output", type=str, default='/data/renruilong/code/scanqa/visdir/scene0389_00/scene0389_00_align_with_pred_gt_box.ply', help='the path of output file')
    parser.add_argument("--gpu", type=str, default='7', help='the path of output file')
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # 可视化
    vis_bbox(scene_ply_path=args.scene_ply,
                scene_txt_path=args.scene_txt,
                bbox_info_path=args.bbox_info,
                mode=args.mode,
                output=args.output)