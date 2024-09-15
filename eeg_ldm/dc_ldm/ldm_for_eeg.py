import numpy as np
import wandb
import torch
from dc_ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import os
from dc_ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sc_mbm.mae_for_eeg import eeg_encoder, classify_network, mapping 
from PIL import Image
from CNN import ConvNet
from scipy.fftpack import fft, rfft, fftfreq, irfft, ifft, rfftfreq
import os

# from PSD_ExtractFeatures import PSD_Etract
# from transformer_model import Transformer



def create_model_from_config(config, num_voxels, global_pool):
    model = eeg_encoder(time_len=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool) 
    return model

def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0

class cond_stage_model(nn.Module):
    def __init__(self, metafile, num_voxels=440, cond_dim=1280, global_pool=True, clip_tune = True, cls_tune = False , device = None):
        super().__init__()
        # prepare pretrained fmri mae 
        if metafile is not None:
            model = create_model_from_config(metafile['config'], num_voxels, global_pool)
        
            model.load_checkpoint(metafile['model'])
        else:
            model = eeg_encoder(time_len=num_voxels, global_pool=global_pool)
        # cnn_class_embedding
        self.cnn_path = '/home/class_cnn_model.ckpt'
        self.num_classes = 40
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        modelCNN = ConvNet(self.num_classes).to(device)
        modelCNN.load_state_dict(torch.load('class_cnn_model.ckpt'))
        self.cnnClass = modelCNN

        # transformer_class_embedding
        from dc_ldm.transformer_model import Transformer
        self.transformer_model = Transformer().cuda()
        self.transformer_path = '/home/ImageNetClass_TransformerEncoder_3382_0.9469_0.8851_0.8992_weights.pth'
        ckpt = torch.load(self.transformer_path)
        self.transformer_model.load_state_dict(ckpt['net'], strict=False)


        self.mae = model
        if clip_tune:
            self.mapping = mapping()
        if cls_tune:
            self.cls_net = classify_network()

        self.fmri_seq_len = model.num_patches
        self.fmri_latent_dim = model.embed_dim
        if global_pool == False:
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
                nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)
            )
            self.channel_class_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
                nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)
            )
        self.dim_mapper = nn.Linear(self.fmri_latent_dim, 767, bias=True)
        self.dim_class_mapper = nn.Linear(1, 1, bias=True)

        self.dim_class_mapper_16 = nn.Linear(4, 1, bias=True)
        self.dim_class_mapper_1 = nn.Linear(1, 1, bias=True)

        self.para_class = nn.Parameter(torch.randn(1,77,1))

        self.global_pool = global_pool
        # self.image_embedder = FrozenImageEmbedder()

    # def forward(self, x):
    #     # n, c, w = x.shape
    #     latent_crossattn = self.mae(x)
    #     if self.global_pool == False:
    #         latent_crossattn = self.channel_mapper(latent_crossattn)
    #     latent_crossattn = self.dim_mapper(latent_crossattn)
    #     out = latent_crossattn
    #     return out


    def forward(self, x, target, fre_eeg, time_eeg):
        # n, c, w = x.shape
        latent_crossattn = self.mae(x)
        latent_return = latent_crossattn

        if latent_crossattn.shape[0] == 4:
            # class_embedding = self.cnnClass(x = None, target = target, device = self.device).expand(4, -1, -1).permute(0, 2, 1)
            # fre_eeg  =  fre_eeg.view(1, fre_eeg.shape[0], fre_eeg.shape[1])
            # time_eeg =  time_eeg.view(1, time_eeg.shape[0], time_eeg.shape[1])
            class_embedding = self.transformer_model(fre_eeg, time_eeg).expand(4, -1, -1).permute(0, 2, 1)
        elif latent_crossattn.shape[0] == 3:
            # class_embedding =  self.cnnClass(x = None,target = target, device = self.device).expand(3, -1, -1).permute(0, 2, 1)
            fre_eeg  =  fre_eeg.view(1, fre_eeg.shape[0], fre_eeg.shape[1])
            time_eeg =  time_eeg.view(1, time_eeg.shape[0], time_eeg.shape[1])
            class_embedding = self.transformer_model(fre_eeg, time_eeg).expand(4, -1, -1).permute(0, 2, 1)
        elif fre_eeg.shape[0] == 1:
            fre_eeg = fre_eeg
            time_eeg = time_eeg
            class_embedding = self.transformer_model(fre_eeg, time_eeg).expand(4, -1, -1).permute(0, 2, 1)
        else:
            # class_embedding = self.cnnClass(x = None,target = target, device = self.device).permute(0, 2, 1)
            fre_eeg  =  fre_eeg.view(1, fre_eeg.shape[0], fre_eeg.shape[1])
            time_eeg =  time_eeg.view(1, time_eeg.shape[0], time_eeg.shape[1])
            class_embedding = self.transformer_model(fre_eeg, time_eeg).expand(4, -1, -1).permute(0, 2, 1)
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_crossattn)
        latent_crossattn = self.dim_mapper(latent_crossattn)

        if self.global_pool == False:
            latent_class_crossattn = self.channel_class_mapper(class_embedding)  #(1,77,1)
        if latent_class_crossattn.shape[2] == 4:
            latent_class_crossattn = self.dim_class_mapper_16(latent_class_crossattn)
        elif latent_crossattn.shape[0] == 1:
            latent_class_crossattn = self.dim_class_mapper(latent_class_crossattn)
        else:
            latent_class_crossattn = self.dim_class_mapper(latent_class_crossattn)

        # if latent_crossattn.shape[0] != latent_class_crossattn.shape[0] and latent_crossattn.shape[1] != latent_class_crossattn.shape[1]:
        if latent_crossattn.shape[0] == 5:
            # latent_class_crossattn =latent_class_crossattn.expand(latent_crossattn.shape[0], -1, -1)
            latent_class_crossattn =torch.cat((latent_class_crossattn, self.para_class), dim = 0)
        elif latent_crossattn.shape[0] == 4:
            latent_class_crossattn = latent_class_crossattn
        elif latent_crossattn.shape[0] == 1:
            latent_class_crossattn = latent_class_crossattn[:1, :, :]
        else:
            latent_class_crossattn = latent_class_crossattn[:3, :, :]
        out = torch.cat((latent_crossattn, latent_class_crossattn), dim=2)

        return out, latent_return

    # def recon(self, x):
    #     recon = self.decoder(x)
    #     return recon

    def get_cls(self, x):
        return self.cls_net(x)

    def get_clip_loss(self, x, image_embeds):
        # image_embeds = self.image_embedder(image_inputs)
        target_emb = self.mapping(x)
        # similarity_matrix = nn.functional.cosine_similarity(target_emb.unsqueeze(1), image_embeds.unsqueeze(0), dim=2)
        # loss = clip_loss(similarity_matrix)
        loss = 1 - torch.cosine_similarity(target_emb, image_embeds, dim=-1).mean()
        return loss
    


class eLDM:

    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=False, clip_tune = True, cls_tune = False):
        # self.ckp_path = os.path.join(pretrain_root, 'model.ckpt')
        # self.ckp_path = os.path.join(pretrain_root, 'models/v1-5-pruned.ckpt')
        # self.config_path = os.path.join(pretrain_root, 'models/config15.yaml')
        self.ckp_path = '/home/v1-5-pruned.ckpt'


        # self.ckp_path = '/home/model.ckpt'
        self.config_path = '/home/config15.yaml'

        #todo class_embedding
        self.cnn_path = '/home/cnn_model.ckpt'
        # class_embedding_model = torch.load(self.class_path)

        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim

        model = instantiate_from_config(config.model)
        pl_sd = torch.load(self.ckp_path, map_location="cpu")['state_dict']
       
        m, u = model.load_state_dict(pl_sd, strict=False)
        model.cond_stage_trainable = True
        model.cond_stage_model = cond_stage_model(metafile, num_voxels, self.cond_dim, global_pool=global_pool, clip_tune = clip_tune,cls_tune = cls_tune, device = device)

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        
        self.device = device    
        self.model = model
        
        self.model.clip_tune = clip_tune
        self.model.cls_tune = cls_tune

        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim
        self.metafile = metafile

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one
      
        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bs1, shuffle=False)
        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()
        # self.model.freeze_whole_model()
        # self.model.unfreeze_cond_stage()

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)

        self.model.unfreeze_whole_model()
        
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'state': torch.random.get_rng_state()

            },
            os.path.join(output_path, 'checkpoint.pth')
        )
        

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path = None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            torch.cuda.set_rng_state(state)
            
        with model.ema_scope():
            model.eval()
            for count, item in enumerate(fmri_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                # print(item)
                latent = item['eeg']
                target = item['label']
                
                fre_eeg = item['fre_eeg']
                time_eeg = item['time_eeg']
                
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w') # h w c
                print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'
                
                c, re_latent = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device),target, fre_eeg, time_eeg)
                # c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                if output_path is not None:
                    samples_t = (255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
                    for copy_idx, img_t in enumerate(samples_t):
                        img_t = rearrange(img_t, 'c h w -> h w c')
                        Image.fromarray(img_t).save(os.path.join(output_path, 
                            f'./test{count}-{copy_idx}.png'))
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)
