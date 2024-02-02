# HyperPlanes: Hypernetwork Approach to Rapid NeRF Adaptation


<p align="center">
    <img src=assets/nerf_exp_1.png>
</p>

### Main code to be added

### Abstract: 
*Neural radiance fields (NeRFs) are a widely accepted standard for synthesizing new 3D object views from a small number of base images. However, NeRFs have limited generalization properties, which means that we need to use significant computational resources to train individual architectures for each item we want to represent. To address this issue, we propose a few-shot training approach based on the hypernetwork paradigm that does not require gradient optimization during inference. The hypernetwork gathers information from the training data and generates an update for universal weights. As a result, we have developed an efficient method for generating a high-quality 3D object representation from a small number of images in a single step. This has been confirmed through direct comparison with the current state-of-the-art and a comprehensive ablation study.*


# Usage

### Data

Download the datasets and put them into `data/` directory.

- #### ShapeNet (128x128) (Original) - From original paper

- #### ShapeNet (200x200) - From our dataset
We use custom dataset adapted from Shapenet, which contains cars, chairs and planes classes. Each class has 50 images of size 200x200 and a corresponding pose for each render.


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
- #### Shapenet 128x128
    ```
    python run_hyperplanes_original.py --config configs/cars_128x128.txt
    ```

- #### ShapeNet 200x200
    ```
    python run_hyperplanes_new.py --config configs/cars_200x200.txt
    ```

### Evaluation
- #### Shapenet 128x128
    ```
    python run_hyperplanes_original_eval.py --config {config path} --checkpoint {checkpoint path} --eval_pretraining_iters {iters}
    ```

- #### ShapeNet 200x200
    ```
    python run_hyperplanes_new_eval.py --config {config path} --checkpoint {checkpoint path} --eval_pretraining_iters {iters}
    ```

# Results
 <p align="center">
    <img src=assets/recons_small.png height=400>
</p>
