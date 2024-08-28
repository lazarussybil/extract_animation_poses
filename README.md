# 3D动画抽取动作序列（Extract 3d animation poses）

关键词：blender, fbx, bpy, openpose, dwpose

魔改自：https://github.com/Knife14/extract-animation-poses

Openpose: https://github.com/Hzzone/pytorch-openpose?tab=readme-ov-file

DWpose: https://github.com/IDEA-Research/DWPose/tree/main

## 使用场景
将 FBX 格式的3D动画用 blender 渲染，对帧画面使用 openpose / dwpose 进行姿势提取。

## 使用教程
### 方法一：Docker 镜像

拉取镜像：docker push lazarussybil/extract_animation_poses:0.0.3

启动指令：docker run -p 8000:7777 --gpus all -it extract_animation_poses_llx:0.0.2 /bin/bash

### 方法二：安装环境
1. 确保 python 3.11 的环境【3.10大概率支持，3.12、3.7不支持】
2. 安装 pytorch cuda cuDNN 环境
3. Pip install -r requirements.txt （注意现在默认的是numpy2，会有warning，可以选择降级）
4. 下载模型，
- Openpose 相关：https://drive.google.com/drive/folders/1JsvI4M4ZTg98fmnCZLFM-3TeovnCRElG  保存在 pytorch_openpose/model/
- DWpose 相关：https://drive.google.com/file/d/1Oy9O18cYk8Dk776DbxpCPWmJtJCl-OCm/view  保存在 dwpose/ckpts/


### 使用指令

```
python .\extract_animation_poses.py -a 'fbx_animation_file_path' -m 'fbx_model_file_path' -o 'output_path' -f int32 -x float -y float -z float --mode [dwpose/openpose] --is_draw [0/1]
```

指令示例：
python extract_animation_poses.py -m C:\Users\Ronson\Downloads\动画测试2\动画测试2\FBX2013\Ch33_nonPBR_Rig.fbx -a C:\Users\Ronson\Downloads\动画测试2\动画测试2\Animation\FBX2013\AnimTest.fbx -o D:\0821\test -f 100 -x -0 -y -6 -z 0.8  --mode dwpose --is_draw 1

参数解释：

-a | --animation_3d： FBX文件，以.fbx结尾
-m | --model_3d： FBX文件，以.fbx结尾
-o | --output_path: 输出文件夹
-f | --frames: 动画帧数，默认60，建议100
-x | --camera_x: blender中摄像头的x坐标
-y | --camera_y: blender中摄像头的y坐标
-z | --camera_z: blender中摄像头的z坐标
--mode: 姿态提取模型，可选值 openpose / dwpose
--is_draw: 是否保存姿态图像，可选值 1 / 0



