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


![V2M4's framework](assets/framework.jpg)

We present V2M4, a novel 4D reconstruction method that **directly generates a usable 4D mesh animation asset from a single monocular video**. Unlike existing approaches that rely on priors from multi-view image and video generation models, our method is based on native 3D mesh generation models. Naively applying 3D mesh generation models to generate a mesh for each frame in a 4D task can lead to issues such as incorrect mesh poses, misalignment of mesh appearance, and inconsistencies in mesh geometry and texture maps. To address these problems, we propose a structured workflow that includes camera search and mesh reposing, condition embedding optimization for mesh appearance refinement, pairwise mesh registration for topology consistency, and global texture map optimization for texture consistency. **Our method outputs high-quality 4D animated assets that are compatible with mainstream graphics and game software.** Experimental results across a variety of animation types and motion amplitudes demonstrate the generalization and effectiveness of our method.

<div align=center><img src="assets/Comparison.jpg" alt="Visual comparisons1"></div>

## Citation & Acknowledgments
If you find this paper useful in your research, please consider citing:
```
@article{chen2025v2m44dmeshanimation,
        title={V2M4: 4D Mesh Animation Reconstruction from a Single Monocular Video},
        author={Chen, Jianqi and Zhang, Biao and Tang, Xiangjun and Wonka, Peter},
        journal={arXiv preprint arXiv:2503.09631},
        year={2025}
}
```

