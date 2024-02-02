# HyperPlanes: Hypernetwork Approach to Rapid NeRF Adaptation


<p align="center">
    <img src=assets/nerf_exp_1.png height=200>
</p>


### Abstract: 
*Neural radiance fields (NeRFs) provide a state-of-the-art quality for synthesizing novel views of 3D objects from a small subset of base images. However, NeRFs are currently severely limited by the lack of generalization properties. Consequently, we must devote considerable computational resources and time to train individual architectures for each 3D object we want to represent. This limitation can be tackled using generative models to produce 3D objects or rely on a few-shot approach to generate NeRF initialization. However, the latter case of implicit representation requires a few thousand iterations in during inference, while the generative models are large and complex to train. To solve these problems, we propose a hypernetwork-based few-shot training approach, which does not require gradient optimization in inference time. Hypernetwork aggregates information from training data and produces an update for universal weights. Our novel solution combines NeRFs with hypernetworks and the partially non-trainable MultiPlaneNeRF representation. As a result, we obtain an efficient method to generate a 3D object representation from an image in a single step, which is confirmed by various experimental results, including a direct comparison with the state-of-the-art and a comprehensive ablation study.*



## Requirements
- CUDA
- Python 3.9.12
- Dependencies stored in `environment.yml`


# Usage

### Data

Download the datasets and put them into `data/` directory.

- #### ShapeNet (128x128) (Original) - From original paper

- #### ShapeNet (200x200) - From our dataset
We use custom dataset adapted from Shapenet, which contains cars, chairs and planes classes. Each class has 50 images with size 200x200 and corelated pose for each render.


### Installation
```
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

after the requirements.txt file, install pytorch with the specified CUDA version.

For example:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


### Running
- #### Neptune
    ```
    export NEPTUNE_API_TOKEN="token"
    ```
- #### Shapenet 128x128
    ```
    python run_hyperplanes_original.py --config configs/cars_128x128.txt
    ```

- #### ShapeNet 200x200
    ```
    python run_hyperplanes_new.py --config configs/cars_200x200.txt
    ```


# Results
 <p align="center">
    <img src=assets/recons_small.png height=400>
</p>
