import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from stereo import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--img-path', type=str)
    # parser.add_argument('--outdir', type=str, default='./vis_depth')
    #parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    
    img_path = './source_images'
    encoder = 'vits'
    outdir = './depth_images'
    # parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    # parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    #args = parser.parse_args()
    
    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    if os.path.isfile(img_path):
        if img_path.endswith('txt'):
            with open(img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [img_path]
    else:
        filenames = os.listdir(img_path)
        filenames = [os.path.join(img_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(outdir, exist_ok=True)
    
    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        left_img = raw_image.copy()
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        h, w = image.shape[:2]
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            depth = depth_anything(image)
        
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
        depth = depth.cpu().numpy().astype(np.uint8)
        
        filename = os.path.basename(filename)        
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        #Write Image to depth_images folder
        cv2.imwrite(os.path.join('depth_images', filename[:filename.rfind('.')] + '_depth.png'), depth)
        
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        depth = cv2.blur(depth, (3, 3))

        if len(depth.shape) == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

        right_img = stereo_gen(left_img, depth)
        stereo = np.hstack([right_img,left_img])

        #Write Image to stereo_images folder
        cv2.imwrite(os.path.join('stereo_images', filename[:filename.rfind('.')] + '_stereo.png'), stereo)

        