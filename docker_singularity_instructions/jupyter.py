# Auto reload and inline plotting
%reload_ext autoreload
%autoreload 2
%matplotlib inline

from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor

import numpy as np

PATH = "/dataset/BSDS300/images/test"
model_filename = "/code/model_epoch_5.pth"
use_cuda = True

os.listdir(PATH)

files = listdir(PATH)[:5]

input_image = PATH + files[0]; 
img = Image.open(input_image).convert('YCbCr')
y, cb, cr = img.split()
display_image = plt.imread(input_image)
plt.imshow(display_image);

img.shape

img[:4, :4]

model = torch.load(model_filename)
img_to_tensor = ToTensor()
input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

if use_cuda:
    model = model.cuda()
    input = input.cuda()

out = model(input)
out = out.cpu()
out_img_y = out[0].detach().numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
plt.imshow(out_img);

output_filename = "/code/out.png"
out_img.save(output_filename)
print('output image saved to ', output_filename)
