from einops import repeat
from models.autoencoder import AutoencoderKL
from models.diffusion import LatentDiffusion
from models.utils.ddim import DDIMSampler
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import autocast



first_config_checkpoint = {'embed_dim': 1, 'n_embed': 2500, 'learning_rate': 4.5e-06, 'ckpt_path': 'YOUR_PATH_TO_AUTOENCODER_CHECKPOINT',
 'ddconfig': {'double_z': True, 'z_channels': 1, 'resolution': 144, 'in_channels': 1, 'out_ch': 1, 'ch': 128, 'ch_mult': [1, 2, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}, 
 'lossconfig': {'disc_start': 50001, 'kl_weight': 1e-06, 'disc_weight': 0.5}}
cond_config_checkpoint = {'embed_dim': 1, 'n_embed': 2500, 'learning_rate': 4.5e-06, 'ckpt_path': 'YOUR_PATH_TO_TERRAINENCODER_CHECKPOINT',
 'ddconfig': {'double_z': True, 'z_channels': 1, 'resolution': 144, 'in_channels': 1, 'out_ch': 1, 'ch': 128, 'ch_mult': [1, 2, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}, 
 'lossconfig': {'disc_start': 50001, 'kl_weight': 1e-06, 'disc_weight': 0.5},
 'first_stage_config': first_config_checkpoint}
UnetConfig = {
    'image_size': 32,
    'in_channels': 1,
    'out_channels': 1,
    'model_channels': 64,
    'attention_resolutions': [ 4, 2, 1 ],
    'num_res_blocks': 2,
    'channel_mult': [ 1, 2, 4],
    'num_heads': 8,
    'use_spatial_transformer': True,
    'transformer_depth': 1,
    'context_dim': 1,
    'use_checkpoint': True,
    'legacy': False
}
LatentConfig_1 = {
    'first_stage_config': first_config_checkpoint,
    'cond_stage_config': cond_config_checkpoint,
    'timesteps': 1000,
    'cosine_s': 8e-3,
    'UnetConfig': UnetConfig,
    'm_id': 1
}
LatentConfig_2 = {
    'first_stage_config': first_config_checkpoint,
    'cond_stage_config': cond_config_checkpoint,
    'timesteps': 1000,
    'cosine_s': 8e-3,
    'UnetConfig': UnetConfig,
    'm_id': 2
}
LatentConfig_3 = {
    'first_stage_config': first_config_checkpoint,
    'cond_stage_config': cond_config_checkpoint,
    'timesteps': 1000,
    'cosine_s': 8e-3,
    'UnetConfig': UnetConfig,
    'm_id': 3
}

def sampling(x, batch_size):
    x_ = torch.tensor(x, dtype=torch.float32,device=torch.device('cuda'))
    if len(x_.shape) == 3:
        x_ = x_[None, ...]
    x_ = x_.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    decoder = AutoencoderKL(**first_config_checkpoint)
    decoder.eval()
    decoder = decoder.to(device)
    ## first denoiser
    pl_sd = torch.load('path_to_your_first_diffusion_checkpoint', map_location="cpu")
    model_1 = LatentDiffusion(**LatentConfig_1)
    sd = pl_sd["state_dict"]
    model_1.load_state_dict(sd, strict=False)
    model_1.cuda()
    model_1.eval()
    model_1 = model_1.to(device)

    ## second denoiser
    pl_sd = torch.load('path_to_your_first_diffusion_checkpoint', map_location="cpu")
    model_2 = LatentDiffusion(**LatentConfig_2)
    sd = pl_sd["state_dict"]
    model_2.load_state_dict(sd, strict=False)
    model_2.cuda()
    model_2.eval()
    model_2 = model_2.to(device)

    ## third denoiser
    pl_sd = torch.load('D:/S2T/lightning_logs/version_71/checkpoints/epoch=99-step=133500.ckpt', map_location="cpu")
    model_3 = LatentDiffusion(**LatentConfig_3)
    sd = pl_sd["state_dict"]
    model_3.load_state_dict(sd, strict=False)
    model_3.cuda()
    model_3.eval()
    model_3 = model_3.to(device)

    c = model_1.get_learned_conditioning(x_)
    init_latent = repeat(c, '1 ... -> b ...', b=batch_size)
    label_image_input = repeat(label_image_input, '1 ... -> b ...', b=batch_size)
    x_ = repeat(x_, '1 ... -> b ...', b=batch_size)
    sampler = DDIMSampler(model_1, model_2, model_3, method='x0')
    sampler.make_schedule(ddim_num_steps=999, ddim_eta=0.0, verbose=False)
    precision_scope = autocast
    sampling_steps = 36
    im_source = np.uint8(x * 255)
    with torch.no_grad():
        with precision_scope("cuda"):
            z_enc = sampler.stochastic_encode(x0 = init_latent, t= torch.tensor([998]*batch_size).to(device))
            samples = sampler.decode(z_enc, x_, sampling_steps)
            x_samples = decoder.decode(samples)
            all_drawing = []
            for b in range(batch_size):
                y_hat = x_samples[b].permute(1, 2, 0).to(memory_format=torch.contiguous_format).float()
                predicted = np.uint8((y_hat.cpu().detach().numpy() * 2 -1) * 127.5 + 127.5)
                all_drawing.append(np.squeeze(predicted, axis=-1))
                im_c = np.concatenate((np.squeeze(predicted, axis=-1), im_source[..., 0], im_source[..., 1], im_source[..., 2], im_source[..., 3]), axis=1)
                plt.imsave(f'./sketch_conversion_terrain_diffusion_sketch_2_{b}.png', im_c, cmap='terrain')
                np.squeeze(predicted, axis=-1).astype('int16').tofile(f'heightmap/heightmap_sketch_2_{b}.raw')

if __name__ == '__main__':
    batch_size = 50 
    input_data = np.load('path_to_your_sketch')
    sampling(batch_size,input_data)