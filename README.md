<div align="center">

<h1><a href="https://arxiv.org/abs/2503.09631">V2M4: 4D Mesh Animation Reconstruction from a Single Monocular Video</a></h1>

**[Jianqi Chen](https://windvchen.github.io/), [Biao Zhang](https://scholar.google.co.uk/citations?user=h5KukxEAAAAJ&hl=en&oi=ao), [Xiangjun Tang](https://scholar.google.co.uk/citations?user=9vZn91sAAAAJ&hl=en&oi=ao), and [Peter Wonka](https://scholar.google.co.uk/citations?user=0EKXSXgAAAAJ&hl=en)**

![](https://komarev.com/ghpvc/?username=windvchenV2M4&label=visitors)
![GitHub stars](https://badgen.net/github/stars/windvchen/V2M4)
[![](https://img.shields.io/badge/license-MIT-blue)](#License)
[![](https://img.shields.io/badge/arXiv-2503.09631-b31b1b.svg)](https://arxiv.org/abs/2503.09631)
<a href='https://windvchen.github.io/V2M4/'>
  <img src='https://img.shields.io/badge/Project-Page-pink?style=flat&logo=Google%20chrome&logoColor=pink'></a>

</div>

![V2M4's preface](assets/teaser.png)

### Share us a :star: if this repo does help

This repository is the official implementation of ***V2M4*** (***Accepted by ICCV 2025***)! 🚀

If you encounter any question about the paper, please feel free to contact us. You can create an issue or just send email to me windvchen@gmail.com. Also welcome for any idea exchange and discussion.


## Updates

[**07/25/2025**] Enhanced the experimental **CoTracker3** tracking feature!  
- Added a new `--tracking_camera_radius` argument to control the camera's distance from the object, helping keep the object within view during motion.  
- Introduced debugging visualizations: tracking videos and tracked result videos are now saved for easy review.  
**How this feature works:** By tracking key points across rendered frames, we regularize mesh registration to keep vertex positions consistent throughout the animation.

[**07/09/2025**] Benchmark Data Released.👋 We have released our evaluation dataset on [Google Drive](https://drive.google.com/drive/folders/1cv-GmODZ0eMGSh_aX8XKuFcRi6BOpUo0?usp=sharing). The benchmark consists of two folders:

- **simple**: Contains 20 animation videos from Consistent4D, featuring objects with simple topologies and subtle movements.
- **complex**: Contains 20 animation videos, each paired with its original 4D file (`.fbx`/`.blend`) sourced from Mixamo and Sketchfab. These samples cover larger-scale object motions and more complex topologies.

[**07/04/2025**] Evaluation code released.

[**06/30/2025**] 🎉 Code is now publicly released! We invite you to try it out. The released version includes several **NEW FEATURES** beyond the initial paper:

- Support for multiple state-of-the-art 3D generators: **TRELLIS**, **Hunyuan3D-2.0**, **TripoSG**, and **CraftsMan3D**
- Integration of advanced stereo prediction techniques: **DUSt3R** and **VGGT** for improved camera search
- Enhanced mesh registration with the **CoTracker3** tracking technique

Explore these new capabilities and let us know your feedback! 🚀🚀🚀

[**06/25/2025**] Paper accepted by **ICCV 2025**. 🎉🎉

[**03/18/2025**] Repository init.


## Table of Contents
- [Abstract](#abstract)
- [Requirements & Installation](#Requirements--Installation)
- [4D Mesh Animation Reconstruction](#4D-Mesh-Animation-Reconstruction)
- [Rendering videos based on the reconstructed results](#rendering-videos-based-on-the-reconstructed-results)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation & Acknowledgments](#citation--acknowledgments)
- [License](#license)

## Abstract

![V2M4's framework](assets/framework.jpg)

We present V2M4, a novel 4D reconstruction method that **directly generates a usable 4D mesh animation asset from a single monocular video**. Unlike existing approaches that rely on priors from multi-view image and video generation models, our method is based on native 3D mesh generation models. Naively applying 3D mesh generation models to generate a mesh for each frame in a 4D task can lead to issues such as incorrect mesh poses, misalignment of mesh appearance, and inconsistencies in mesh geometry and texture maps. To address these problems, we propose a structured workflow that includes camera search and mesh reposing, condition embedding optimization for mesh appearance refinement, pairwise mesh registration for topology consistency, and global texture map optimization for texture consistency. **Our method outputs high-quality 4D animated assets that are compatible with mainstream graphics and game software.** Experimental results across a variety of animation types and motion amplitudes demonstrate the generalization and effectiveness of our method.

## Requirements & Installation

1. Hardware Requirements
    - GPU: 1x high-end NVIDIA GPU with at least 40GB memory

2. Installation
   - Please follow the detailed instructions in [Install.md](Install.md) to install the required packages and set up the environment.
   - The code has been tested with **Python 3.10 + Pytorch 2.4.0 + CUDA 12.1**.

3. Datasets
   - There have been demo-datasets in [examples](examples), you can directly run the reconstruction code below to see the results. For the full benchmark dataset used in our paper, please access [Google Drive](https://drive.google.com/drive/folders/1cv-GmODZ0eMGSh_aX8XKuFcRi6BOpUo0?usp=sharing) to download the evaluation dataset.
   - If you want to test your own videos, please follow the format of the demo datasets. (The input image frames could be either RGB images with background or transparent RGBA images.) We also provide some useful data preparation scripts in [data_preparation_tools](data_preparation_tools) to help you prepare the input data.
  *(For better performance, it is **highly recommended** that **the object in the first frame has minimal part overlap and that no parts are touching each other**. This helps avoid artificial topology issues during reconstruction.)*

## 4D Mesh Animation Reconstruction
To reconstruct a 4D mesh animation from a single monocular video, please refer to the following command:

```bash
python main.py \
  --root {your_video_folder} \
  --output {your_output_folder} \
  --model Hunyuan \  # Performance order in our experiments (from good to bad): Hunyuan ≈ TripoSG > TRELLIS ≈ Craftsman3D. (Actual performance may vary depending on your data and use case.)
  --N 1 \
  --n 0 \
  --skip 5 \
  --seed 42 \
  --use_vggt         # (Highly Recommend) Use VGGT for camera search; omit for USING DUSt3R
  --baseline         # (optional) Run the baseline model, i.e., directly use the 3D mesh generator to generate a mesh for each frame without V2M4
  --use_tracking     # (experimental) Use point tracking for mesh registration guidance, will add more memory usage and time cost
  --tracking_camera_radius {radius}  # (experimental) Set the camera tracking radius (this is only valid when using `--use_tracking`) to keep the object visible during motion, default is 8
  --blender_path {your_blender_path}  # Directory path of Blender executable
```

**Argument Descriptions:**
- `--root`: Root directory of the dataset
- `--output`: Output directory for results (there will be quite detailed intermediate results saved in this folder for debugging and analysis, the final reconstructed glb file's name will be `output_animation.glb`)
- `--model`: Base model to use `TRELLIS`, `Hunyuan`, `TripoSG`, or `Craftsman` (Performance order in our experiments: Hunyuan ≈ TripoSG > TRELLIS ≈ Craftsman3D.)
- `--N`: Total number of parallel processes (default: 1)
- `--n`: Index of the current process (default: 0)
- `--skip`: Skip every N frames for large object movement (default: 5)
- `--seed`: Random seed for reproducibility (default: 42)
- `--baseline`: Run the baseline model (flag)
- `--use_vggt`: Use VGGT for camera search (omit for USING DUSt3R)
- `--use_tracking`: Use point tracking for mesh registration guidance
- `--tracking_camera_radius`: Set the camera tracking radius to keep the object visible during motion (default: 8, only valid when using `--use_tracking`)
- `--blender_path`: Path to Blender executable (example: `blender-4.2.1-linux-x64/`)

**Example:**
```bash
python main.py --root examples --output results --model Hunyuan --N 1 --n 0 --skip 5 --seed 42 --use_vggt --use_tracking --tracking_camera_radius 8 --blender_path blender-4.2.1-linux-x64/
```

***Note1:** In some cases, the reconstruction results may not be satisfactory. We recommend experimenting with different random seeds and adjusting the `--skip` value (for videos with more intense motion, use a smaller `--skip` value) to potentially achieve better outcomes.*

***Note2:** If the object disappears from view during animation, it may be because the camera tracking radius is too small, causing the object to move out of frame when rendering from multiple views. To debug, check the saved tracking videos. You can fix this by increasing the `--tracking_camera_radius` value to keep the object visible, or by disabling tracking (remove `--use_tracking`).*

## Rendering videos based on the reconstructed results 

After reconstructing your 4D mesh animation, you can render videos from the generated mesh sequences using our provided [script](rendering_video.py). This script will render images and videos from the reconstructed `.glb` mesh files for each animation.

**Usage:**

```bash
python rendering_video.py --result_path {your_results_folder} [--baseline] [--normal] [--interpolate N]
```

**Argument Descriptions:**
- `--result_path`: Path to the folder containing your reconstructed results (default: `results`)
- `--baseline`: (Optional) Also render videos for the baseline mesh results
- `--normal`: (Optional) Render normal maps in addition to texture images
- `--interpolate N`: (Optional) Number of interpolation steps between frames for smoother animation (default: 1)

**Example:**

```bash
python rendering_video.py --result_path results --baseline --normal
```

**Output:**
This will generate the following outputs for each animation:
- Individual rendered images in `output_final_rendering_images/` subfolder
- Main animation video: `output_final_rendering_video.mp4`
- Interpolated animation video: `output_final_rendering_video_interpolated_{N}.mp4` (if `--interpolate` > 1)
- Normal map videos (if `--normal` flag is used)
- Baseline comparison videos (if `--baseline` flag is used)

**Requirements:**
- Your results folder should contain the reconstructed `.glb` files with `_texture_consistency_sample.glb` suffix
- An `extrinsics_list.pkl` file for each animation (automatically generated during reconstruction)
- If using `--baseline`, files with `_baseline_sample.glb` suffix should also be present

## Evaluation
To evaluate the quality of animation reconstruction results against ground truth videos, we provide a comprehensive evaluation script that calculates several widely used video similarity metrics.

The evaluation includes the following metrics: **FVD**, **LPIPS** , **DreamSim**, and **CLIP Loss**. 

```bash
cd evaluation
python evaluation.py --gt_videos_path {path_to_GT_videos} --result_videos_path {path_to_V2M4_rendering_videos}
```

Please ensure your video structure follows this format:
```
├── path_to_GT_videos/
│   ├── animation1.mp4
│   ├── animation2.mp4
│   └── ...
├── path_to_V2M4_rendering_videos/
│   ├── animation1.mp4
│   ├── animation2.mp4
│   └── ...
```

## Results

The results below were obtained using the TRELLIS generator with DUSt3R for camera search, as described in our initial paper. For improved performance, we recommend trying our newly supported models, such as Hunyuan3D-2.0 and VGGT.

<div align=center><img src="assets/Comparison.jpg" alt="Visual comparisons1"></div>

## Citation & Acknowledgments
If you find this paper useful in your research, please consider citing:
```
@article{chen2025v2m4,
  title={V2M4: 4D Mesh Animation Reconstruction from a Single Monocular Video},
  author={Chen, Jianqi and Zhang, Biao and Tang, Xiangjun and Wonka, Peter},
  journal={arXiv preprint arXiv:2503.09631},
  year={2025}
}
```

We gratefully acknowledge the authors and contributors of the following open-source projects, whose work made this research possible: [TRELLIS](https://github.com/microsoft/TRELLIS), [CraftsMan3D](https://github.com/wyysf-98/CraftsMan3D), [TripoSG](https://github.com/VAST-AI-Research/TripoSG), [Hunyuan3D-2.0](https://github.com/Tencent-Hunyuan/Hunyuan3D-2), [DUSt3R](https://github.com/naver/dust3r), [VGGT](https://github.com/facebookresearch/vggt), [CoTracker3](https://github.com/facebookresearch/co-tracker), etc. We appreciate their commitment to open research and the broader scientific community.

## License
This project is licensed under the MIT license. See [LICENSE](LICENSE) for details.