# -*- coding: utf-8 -*- #

# -----------------------------
# Topic: Intelligently transforms a animation from 3d to 2d
# Author: motm14
# Created: 2023.04.29
# Description: by blender and openpose, transforms 3d animation to 2d animation
# History:
#    <autohr>    <version>    <time>        <desc>
#    motm14         v0.1    2023/04/30      basic build
# -----------------------------


import bpy
import cv2
import numpy as np
import torch

import argparse
import math
import os
import shutil
import datetime

from pytorch_openpose.src import util


def clear_scene():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

def load_model_animation(model_fbx_path, animation_fbx_path):
    # model_fbx_path = r'D:\0821\测试动画\SK_Mannequin_Sequence.FBX'
    # animation_fbx_path = r'D:\0821\测试动画\Animation\Talking_Casual_HandGesture18.FBX'

    # 导入模型
    bpy.ops.import_scene.fbx(filepath=model_fbx_path, automatic_bone_orientation=True)
    model = bpy.context.selected_objects[0]
    model.name = 'root'

    bpy.ops.import_scene.fbx(filepath=animation_fbx_path, automatic_bone_orientation=True)
    obj = bpy.context.selected_objects[0]   # 获取刚导入的动画
    obj.animation_data.action.name = "test"

    model = bpy.data.objects['root']
    model.animation_data_create()  # 确保animation_data不为空
    model.animation_data.action = bpy.data.actions['test']

def setup_camera(camera_x, camera_y, camera_z):
    if not bpy.data.cameras:
        bpy.ops.object.camera_add()
    camera = bpy.context.scene.camera
    camera.location = (camera_x, camera_y, camera_z)
    camera.rotation_euler = (math.radians(90), math.radians(0), math.radians(0))


def render_animation(model_fbx_path, animation_fbx_path, output, camera_x, camera_y, camera_z, frames=60):
    clear_scene()

    # 导入fbx文件并设定动作
    load_model_animation(model_fbx_path=model_fbx_path, animation_fbx_path=animation_fbx_path)

    # 设定camera
    setup_camera(camera_x=camera_x, camera_y=camera_y, camera_z=camera_z)

    # 设置渲染输出路径和格式
    bpy.context.scene.render.filepath = output
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    
    # 设置帧范围
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frames

    # 设置渲染引擎为Cycles
    # bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'  
    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    # bpy.context.scene.render.engine = 'CYCLES'    ### 这个mode很慢

    # 设置分辨率
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    # bpy.context.scene.render.resolution_x = 960
    # bpy.context.scene.render.resolution_y = 540

    # # 添加渲染队列
    # for obj in bpy.context.scene.objects:
    #     bpy.context.scene.collection.objects.link(obj)
    # bpy.ops.transform.translate(value=(0, 0, 1), orient_type='GLOBAL')

    # 渲染
    bpy.ops.render.render(animation=True)

from pytorch_openpose.src.hand import Hand
from pytorch_openpose.src.body import Body

body_estimation = Body('./pytorch_openpose/model/body_pose_model.pth')
hand_estimation = Hand('./pytorch_openpose/model/hand_pose_model.pth')
def extract_poses(blender_folder, output_folder):
    
    print(f"Torch device: {torch.cuda.get_device_name()}")
    
    file_list = [os.path.join(blender_folder, f) for f in os.listdir(blender_folder) if f.endswith('png')]


    body_estimation = Body('./pytorch_openpose/model/body_pose_model.pth')
    hand_estimation = Hand('./pytorch_openpose/model/hand_pose_model.pth')


    ### ! 取消注释可以获得输出图像相关的代码
    for file_path in file_list:
        oriImg = cv2.imread(file_path)
        img_height, img_width, _ = cv2.imread(file_list[0]).shape

        candidate, subset = body_estimation(oriImg)
        # canvas = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        # canvas = util.draw_bodypose(canvas, candidate, subset)

        hands_list = util.handDetect(candidate, subset, oriImg)
        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            all_hand_peaks.append(peaks)
        # canvas = util.draw_handpose(canvas, all_hand_peaks)

        data = {
            'body' : {
                'candidates': candidate, #身体关节坐标
                'subsets': subset #身体关节连接信息
            },
            'hands': all_hand_peaks
        }

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        output_npy = os.path.join(output_folder, file_path.split('/')[-1].split('.')[0] + '.npy')
        print("Saved: '", output_npy, "'")
        np.save(output_npy, data)

        # output_png = output_folder + file_path.split('/')[-1].split('.')[0] + '.png'
        # print("Saved: '", output_png, "'")
        # cv2.imwrite(output_png, canvas)

    # 释放资源并关闭窗口
    cv2.destroyAllWindows()


def stitch_pngs(prompt_folder, output_folder):
    # 设置文件路径
    file_list = [os.path.join(prompt_folder, f) for f in os.listdir(prompt_folder) if f.endswith('png')]
    
    # 设置画布
    img_height, img_width, _ = cv2.imread(file_list[0]).shape
    canvas_width = img_width * len(file_list)
    canvas = np.zeros((img_height, canvas_width, 3), dtype=np.uint8)
    
    # 拼接
    for i, file_path in enumerate(file_list):
        img = cv2.imread(file_path)
        canvas[:, i*img_width:(i+1)*img_width, :] = img

    # 保存并释放资源
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cv2.imwrite(output_folder + '/' + output_folder.split('/')[-1] + '.png', canvas)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model_3d", type=str,
                        required=True, help="3d model file")
    parser.add_argument('-a', "--animation_3d", type=str,
                        required=True, help="3d animation file")
    parser.add_argument("-o", "--output_path", type=str,
                        required=True, help="output file abs path")
    parser.add_argument("-x", "--camera_x", type=float,
                        required=True, help="x index of camera")
    parser.add_argument("-y", "--camera_y", type=float,
                        required=True, help="y index of camera")
    parser.add_argument("-z", "--camera_z", type=float,
                        required=True, help="z index of camera")
    parser.add_argument("-f", "--frames", type=int,
                        required=True, help="num of frames")
    parser.add_argument("--mode", type=str,
                        required=True, help="pose parsing model")
    args = parser.parse_args()
    
    model_file = args.model_3d
    animation_file = args.animation_3d
    pose_parsing_model = args.mode
    output_path = os.path.join(args.output_path, os.path.basename(animation_file).split('.')[0])
    blender_folder = output_path + '/blender/'
    prompt_folder = output_path + '/prompts/'
    stitch_folder = output_path
    
    # Blender 渲染 
    ### Maya 渲染
    render_animation(model_fbx_path=model_file, animation_fbx_path=animation_file, output=blender_folder, camera_x=args.camera_x, camera_z=args.camera_z, camera_y=args.camera_y, frames=args.frames)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ': render animation successfullly')
    
    
    
    # Openpose/dwpose 提取姿势
    extract_poses(blender_folder, prompt_folder, pose_parsing_model)    

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ': extract poses successfullly')
    
    # # 拼接图片
    # stitch_pngs(prompt_folder, stitch_folder)
    # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ': stitch pngs successfullly')
    
    # 清空渲染、提取姿势文件夹
    shutil.rmtree(blender_folder)
    shutil.rmtree(prompt_folder)
