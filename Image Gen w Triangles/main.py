import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="7"

from datetime import datetime
import os

import numpy as np
import torch
from IPython.display import display
import nvdiffrast.torch as dr
from PIL import Image
import clip

h, w = 256, 256
n_triangles = 50
base_lr = 0.03
report_iterations = 10
n_iterations = 2000

glctx = dr.RasterizeGLContext()

verts = (torch.rand(n_triangles, 3, 2) * 2 - 1).cuda()
verts.requires_grad = True
color = torch.rand(n_triangles, 4).float().cuda()
color.requires_grad = True

def normalize(t, low=0.0, high=1.0):
    # result = ( t - t.min() ) / ( t.max() - t.min() )
    # result = torch.nn.functional.hardsigmoid(t)
    result = torch.nn.functional.sigmoid(t)
    result = result * (high - low) + low
    return result

def normalize_verts(verts):
    low, high = -1.0, 1.0
    ts = [
        normalize(verts[..., ch] * 1.5, low, high)
        for ch in range(2)
    ]
    return torch.stack(ts, dim=-1)

def normalize_color(color):
    low, high = 0.0, 1.0
    ts = [
        normalize(color[..., ch] * 1.5, low, high)
        for ch in range(4)
    ]
    return torch.stack(ts, dim=-1)

faces = torch.arange(n_triangles * 3).reshape(-1, 3).int().cuda()
def combine_layers(out, bg_noise=0.05):
    canvas = torch.ones_like(out[0:1,:,:,0:3])
    canvas = canvas + torch.randn_like(out[0:1,:,:,0:3]) * bg_noise
    for i in range(0, n_triangles):
        alpha = out[i:i+1, ..., 3:4]
        #if i > 150:
        alpha = alpha * 0.1
        draw_color = out[i:i+1, ..., 0:3]
        canvas = canvas * (1 - alpha) + alpha * draw_color
    return canvas


def render(v, c, combine_layers_kwargs={}):
    #verts_norm = v 
    #color_norm = c 
    verts_norm = normalize_verts(v)
    color_norm = normalize_color(c) 
    m_ones = -torch.ones(n_triangles, 3, 1).cuda().float()
    ones = torch.ones(n_triangles, 3, 1).cuda().float()
    verts_in = torch.cat([verts_norm, m_ones, ones], dim =2) # [..., None, :]
    colors = color_norm[:, None, :].repeat(1, 3, 1)
    rast, _ = dr.rasterize(glctx, verts_in, faces, resolution=[h, w], grad_db=True)
    out_inter, _ = dr.interpolate(colors, rast, faces)
    out_layers = dr.antialias(out_inter, rast, verts_in, faces, pos_gradient_boost=1)
    out = combine_layers(out_layers, **combine_layers_kwargs)
    return out


class OptimRateScheduler(object):
    def __init__(self, total_iterations):
        self.total_iterations = total_iterations
        self.ramp_up_to = 1 / 20
        self.ramp_down_from = 3 / 4
    def get_lr_scale(self, c_iter):
        t = c_iter / self.total_iterations
        lr_ramp = min(1.0, (1.0 - t) / 0.25)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / 0.05)
        return lr_ramp
    def get_noise_scale(self, c_iter):
        t = c_iter / self.total_iterations
        if t > self.ramp_down_from:
            return 0
        else:
            return ((self.ramp_down_from - t) / self.ramp_down_from)**2

optim = torch.optim.Adam([verts, color], lr = base_lr)
ors = OptimRateScheduler(total_iterations=n_iterations)

def save_as_gif(fn, imgs, fps=12):
    img, *imgs = imgs
    with open(fn, 'wb') as fp_out:
        img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=int(1000./fps), loop=0)
        
def save_as_frames(fn, imgs, overwrite=True):
    # save to folder `fn` with sequenced filenames
    os.makedirs(fn, exist_ok=True)
    for i, img in enumerate(imgs):
        this_fn = os.path.join(fn, f'{i:08}.png')
        if overwrite or not os.path.exists(this_fn):
            save_as_png(this_fn, img)

def save_as_png(fn, img):
    if not fn.endswith('.png'):
        fn = f'{fn}.png'
    img.save(fn)
            
def save_info_list(fn, info_list):
    with open(fn, 'w') as fout:
        list(map(lambda r: print(r, file=fout), info_list))

import torchvision
import torchvision.transforms as transforms

trans = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

prompt = 'Walt Disney World'


save_dir = f'diff_clip_nvdiffrast_out/[{prompt}]'
os.makedirs(save_dir, exist_ok=True)

device = 'cuda'
model, preprocess = clip.load("ViT-B/32", device=device)

with torch.no_grad():
    text_features = model.encode_text(clip.tokenize(prompt).to(device))

def to_pil_img(img):
    return Image.fromarray((img[0].clip(0,1).detach().cpu().numpy()*255).astype('uint8'))

def show(img):
    display(to_pil_img(img))

def get_loss(out):
    img = out.permute(0, 3, 1, 2)
    t = img
    NUM_AUGS = 10
    t = t.repeat_interleave(NUM_AUGS, dim=0)
    img_augs = trans(t)
    image_features = model.encode_image(img_augs)
    similiarities = torch.cosine_similarity(image_features, text_features, axis=-1)
    loss = -similiarities.mean()
    return loss

pil_img_list = []
info_list = []

for i in range(n_iterations):
    lr = ors.get_lr_scale(i) * base_lr
    optim.param_groups[0]['lr'] = lr
    noise_factor = 0.05
    noise_scale = (noise_factor * ors.get_noise_scale(i))
    verts_noise = torch.randn_like(verts) * noise_scale
    color_noise = torch.randn_like(color) * noise_scale * 0.5
    verts_in = verts + verts_noise
    color_in = color + color_noise
    out = render(verts, color_in, {'bg_noise': 0.05})
    loss = get_loss(out)
    
    optim.zero_grad()
    loss.backward()
    
    if (i + 1) % report_iterations == 0:
        
        with torch.no_grad():
            verts_in = verts
            color_in = color
            out = render(verts, color_in, {'bg_noise': 0.0})
            wonoise_loss = get_loss(out)

        info = f"[{datetime.now()}]   Iteration {i + 1}, lr {optim.param_groups[0]['lr']}, loss {loss.item()} loss (without noise) {wonoise_loss.item()}"
        print(info)
        info_list.append(info)
        save_info_list(f'{save_dir}/log.txt', info_list)
            
        show(out)
        pil_img_list.append(to_pil_img(out))
        save_as_gif(f'{save_dir}/animate.gif', pil_img_list, fps=12)
        save_as_frames(f'{save_dir}/animate.frames', pil_img_list)
        
    optim.step()
