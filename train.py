from data.terraindata import TerrainDataModule
from models.terrainencoder import terrainEncoder
from models.autoencoder import AutoencoderKL
from models.diffusion import LatentDiffusion
import numpy as np
import pytorch_lightning as pl

## Configs, to be cleaned up later
first_config = {'embed_dim': 1, 'n_embed': 2500, 'learning_rate': 4.5e-06,
 'ddconfig': {'double_z': True, 'z_channels': 1, 'resolution': 144, 'in_channels': 1, 'out_ch': 1, 'ch': 128, 'ch_mult': [1, 2, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}, 
 'lossconfig': {'disc_start': 50001, 'kl_weight': 1e-06, 'disc_weight': 0.5}}
cond_config = {'embed_dim': 1, 'n_embed': 2500, 'learning_rate': 4.5e-06,
 'ddconfig': {'double_z': True, 'z_channels': 1, 'resolution': 144, 'in_channels': 1, 'out_ch': 1, 'ch': 128, 'ch_mult': [1, 2, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}, 
 'lossconfig': {'disc_start': 50001, 'kl_weight': 1e-06, 'disc_weight': 0.5},
 'first_stage_config': first_config}
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

def training(data_path:str, first_config:dict, cond_config:dict, LatentConfig_1:dict, LatentConfig_2:dict, LatentConfig_3:dict):
    """
    Training wrapper for models

    :param data_path: path to the training data
    :param first_config: autoencoder config
    :param cond_config: conditional encoder config
    :param LatentConfig_1: config for first diffusion
    :param LatentConfig_1: config for second diffusion
    :param LatentConfig_1: config for third diffusion
    """
    data = np.load(data_path)
    input_image = data['x']
    label_image = data['y']
    dataset = TerrainDataModule(input_image, label_image)
    dataset.setup()
    TE_model = terrainEncoder(**cond_config)
    AEC_model = AutoencoderKL(**first_config)
    Diffusion_model1 = LatentDiffusion(**LatentConfig_1)
    Diffusion_model2 = LatentDiffusion(**LatentConfig_2)
    Diffusion_model3 = LatentDiffusion(**LatentConfig_3)
    cond_trainer = pl.Trainer(accelerator='gpu', default_root_dir="path_to_your_checkpoint_folder")
    cond_trainer.fit(TE_model, dataset)
    AEC_trainer = pl.Trainer(accelerator='gpu', default_root_dir="path_to_your_checkpoint_folder")
    AEC_trainer.fit(AEC_model, dataset)
    DF1_trainer = pl.Trainer(accelerator='gpu', default_root_dir="path_to_your_checkpoint_folder")
    DF1_trainer.fit(Diffusion_model1, dataset)
    DF2_trainer = pl.Trainer(accelerator='gpu', default_root_dir="path_to_your_checkpoint_folder")
    DF2_trainer.fit(Diffusion_model2, dataset)
    DF3_trainer = pl.Trainer(accelerator='gpu', default_root_dir="path_to_your_checkpoint_folder")
    DF3_trainer.fit(Diffusion_model3, dataset)

if __name__ == "__main__":
    training('path_to_your_data',first_config, cond_config, LatentConfig_1, LatentConfig_2, LatentConfig_3)