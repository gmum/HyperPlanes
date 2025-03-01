import torch
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import cv2
import tempfile
import os 
from lpips import LPIPS

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def img2mse(image1, image2) -> float:
    if not torch.is_tensor(image1):
        image1 = torch.tensor(image1)
    if not torch.is_tensor(image2):
        image2 = torch.tensor(image2) 

    image1 = image1.cuda()
    image2 = image2.cuda()
    mse = torch.mean((image1 - image2) ** 2) 
    return mse

def auxilary_ssim(image1, image2):

    image1 = Image.fromarray(to8b(image1))
    image2 = Image.fromarray(to8b(image2))

    image1 = image1.convert("RGB")
    image2 = image2.convert("RGB")
   
    with tempfile.TemporaryDirectory() as tmpdir:
        image1path = os.path.join(tmpdir, "image1.png")
        image2path = os.path.join(tmpdir, "image2.png")
        image1.save(image1path)
        image2.save(image2path)
        image1 = cv2.imread(image1path)
        image2 = cv2.imread(image2path)

    grayA = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(grayA, grayB, channel_axis = 1)
    return ssim_value

def calculate_ssim(image1, image2, logger) -> float:

    if torch.is_tensor(image1):
        image1 = image1.detach().cpu().numpy()
    if torch.is_tensor(image2):
        image2 = image2.detach().cpu().numpy()
    
    assert image1.shape == image2.shape, "images have not even shape" 

    if len(image1.shape) == 2:
        ssim = auxilary_ssim(image1, image2)
        return ssim

    elif len(image1.shape) == 4:
        ssim_storage = []
        for i in range(image1.shape[0]):
            ssim = auxilary_ssim(image1[i], image2[i])
            logger["metrics/single_ssim"].append(ssim)
            ssim_storage.append(ssim)
        ssim_mean = np.mean(ssim_storage)
        return ssim_mean

def calculate_lpips(image1, image2, logger):
    lpips_vgg = LPIPS(net="vgg").to("cuda")
    lpips_storage = []
    if not torch.is_tensor(image1):
        image1 = torch.tensor(image1)
    if not torch.is_tensor(image2):
        image1 = torch.tensor(image2)
    for img1, img2 in zip(image1, image2):
        img1 = img1.permute(2, 0, 1).to("cuda")
        img2 = img2.permute(2, 0, 1).to("cuda")
        # image1 = torch.tensor(image1).permute(0,3,1,2).to("cuda")
        # image2 = torch.tensor(image2).permute(0,3,1,2).to("cuda")
        
        lpips_distance = lpips_vgg(img1, img2).item()
        logger["metrics/single_lpips"].append(lpips_distance)
        lpips_storage.append(lpips_distance)
    
    assert image1.shape == image2.shape, "images have not even shape"
    
    lpips_distance = np.mean(lpips_storage)

    return lpips_distance

