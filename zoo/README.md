# Running other models in zoo  
1. Clone the repos:   
'''bash
git submodule init
git submodule update
'''
2. Run init.sh: 
'''bash
cd zoo
init.sh
'''
3. Download ckpts and perform some special operations on certain specific repos  

## StereoAnything  
Download the `stereoanything.pt` in [here](https://drive.google.com/file/d/18BBk2y7f86PgiEBij3SlCSBwDIurp0K7/view?usp=sharing) and place it at zoo/ckpts/  

Then compile `MultiScaleDeformableAttention`:  
```bash
cd zoo/OpenStereo/stereo/modeling/models/nmrf/ops
python setup.py install
```

Find the file zoo/OpenStereo/stereo/modeling/models/nmrf/backbone.py, pay attention to lines 189~197. If you only want to perform inference, change `pretrained = True` in line 189 to `pretrained = False`. In addition, you can also download the SwinT's pretrained parameter [here](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth) and change the path in line 191 to the path where you place the parameter file.  


# 关于OpenStereo  
包含了IGEV, NMRF等许多双目方法的实现和参数.  
Stereo是基于NMRF做的训练，但不知为何相关的文档说明被某次commit删除了，以下是一些关键的内容：  
* **2024-12-3:** Checkpoint of Stereo Anything is [here](https://drive.google.com/file/d/18BBk2y7f86PgiEBij3SlCSBwDIurp0K7/view?usp=sharing)  
* **2024-11-26:** Code of [Stereo Anything](https://github.com/XiandaGuo/OpenStereo/cfgs/nerf) is released.  
* **2024-11-14:** Stereo Anything: Unifying Stereo Matching with Large-Scale Mixed Data, [*Paper*](https://arxiv.org/abs/2411.14053).  

Here we compare our Stereo Anything with the previous best model.
| Method               | K12   | K15   | Midd  | E3D   | DR    | Mean  |
|--------|-------|---------|-------|-------|-------|-------|
| PSMNet              | 30.51  | 32.15 | 33.53 | 18.02 | 36.19 | 30.08 |
| CFNet               | 13.64 | 12.09 | 23.91 |  7.67 | 27.26 | 16.91 |
| GwcNet              | 23.05 | 25.19 | 29.87 | 14.54 | 35.40 | 25.61 |
| COEX                | 12.08 | 11.01 | 25.17 | 11.43 | 24.17 | 16.77 |
| FADNet++            | 11.31 | 13.23 | 24.07 | 22.48 | 20.50 | 18.32 |
| Cascade             | 11.86 | 12.06 | 27.39 | 11.62 | 27.65 | 18.12 |
| LightStereo-L       |  6.41 |  6.40 | 17.51 | 11.33 | 21.74 | 12.68 |
| IGEV                |  4.88 |  5.16 |  8.47 |  3.53 |  **6.20** |  5.67 |
| StereoBase          |  4.85 |  5.35 |  9.76 |  3.12 | 11.84 |  6.98 |
| NMRFStereo          |  **4.20** |  5.10 |  7.50 |  3.80 | 11.92 |  6.50 |
| NMRFStereo*      |  8.67 |  7.46 | 16.36 | 23.46 | 34.58 | 18.11 |
| **StereoAnything**  |  4.29 |  **4.31** |  **6.96** |  **1.84** |  7.64 |  **5.01** |
We highlight the **best** results in **bold** (**better results**: $\downarrow$).  

You can easily load our pre-trained models by:
```python
python tools/infer.py --cfg_file cfgs/nmrf/nmrf_swint_sceneflow.yaml \
--pretrained_model Your_model_path \
--left_img_path your/path \
--right_img_path your/path \
--savename your/path
```