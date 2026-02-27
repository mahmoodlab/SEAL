"""
Initiates 2D/3D feature extractor, all available with pretrained weights

The current options are
- Resnet50 (2D): The original Resnet, pretrained on Imagenet. 
    - Latent dim: 1024
- UNI (2D)
    - Latent dim: 1024
    - Referece: Chen, Richard J., et al. "Towards a general-purpose foundation model for computational pathology." Nature Medicine 30.3 (2024): 850-862.
- CONCH (2D)
    - Latent dim: 512
    - Referece [to be included]
- CTransPath (2D): 
    - Latent dim: 768
    - Reference: [to be included]
"""
from __future__ import annotations

from torch.utils.data import DataLoader
import re
import json
from peft import PeftModel
from seal.models.components import create_mlp
from torchvision import transforms
from torchvision.transforms import Compose
from .adapter import VisionAdapter

import types
import os
from seal.models.gene_model import GeneMLP, SCGPTEncoder, unfreeze_scgpt, ModelDummy, GeneVAE, GeneTransformer
import torch.nn as nn
from seal.models.da_model import AdversarialDiscriminator
from seal.models.encoder_factory import encoder_factory
from pathlib import Path

HF_API_KEY = os.getenv("HF_API_KEY")

def find_organ_ids():
    rel_path = "cache/organ_ids.json"
    id_file = Path(rel_path)

    if not id_file.exists():
        id_file = Path(__file__).resolve().parent.parent / rel_path
        if not id_file.exists():
            raise FileNotFoundError(f'Could not locate {rel_path} locally or in package.')
    
    with open(id_file, 'r') as f: 
        return json.load(f) # You can return this directly without assigning it to a variable

class ModelMixin(): 
    
    def get_da_model(self, z_dim: int, train_loader: DataLoader):
        """
        Get domain adaptation model for batch correction
        """
        
        if hasattr(train_loader.dataset, 'data_df'): 
            n_cls = train_loader.dataset.data_df['batch_id'].nunique()
        else: 
            n_cls = 2 # dummy
        
        model = AdversarialDiscriminator(
            z_dim=z_dim, # conf['out_dim']
            n_cls=n_cls,
            nlayers=3,
            activation=nn.LeakyReLU,
            reverse_grad=True,
        )

        return model 


    def forward_tokens_2D(self, model, x):
        """
        Return tokens for 2D patch features with CLS tokens removed 
        """
        output = model.forward_features(x)    # (b, L+1, D)

        return output[:, 1:, :]


    def forward_feats_2D(self, model, x):
        return model.forward_features(x)[:, 0, :]


    def forward_conch_feats_2D(self, model, x):
        return model.forward_no_head(x)
        

    def get_gene_model(self, num_genes: int): 
        if self.conf['gene_model'] == "mlp": 
            gene_model = GeneMLP(in_dim=num_genes,
                            out_dim=self.conf['out_dim'], 
                            hidden_dims=self.conf['hidden_dims']
                            )
        elif self.conf['gene_model'] == 'mae': 
            # autoencoder with masking
            gene_model = GeneMLP(in_dim=num_genes,
                                out_dim=self.conf['out_dim'], 
                                hidden_dims=self.conf['hidden_dims'], 
                                mask_ratio=0.2
                                )
        elif self.conf['gene_model'] == 'vae': 
            gene_model = GeneVAE(in_dim=num_genes, 
                                 latent_dim=self.conf['out_dim'],
                                 hidden_dims=self.conf['hidden_dims'], 
                                 mask_ratio=self.conf['mask_ratio'],
                                 dropout=self.conf['vae_dropout'],
                                 projection_head=self.conf['projection_head'],
                                 dec_batch_norm=self.conf['dec_batch_norm'],
                                 dec_dropout=self.conf['dec_dropout'],
                                 )
        
        elif self.conf['gene_model'] == 'transformer':
            gene_model = GeneTransformer(in_dim=num_genes, 
                                         latent_dim=self.conf['out_dim'])
        
        
        
        elif self.conf['gene_model'] is None: 
            gene_model = ModelDummy()

        elif self.conf['gene_model'] == 'scgpt': 
            gene_model = SCGPTEncoder(pretrained=self.conf['scgpt_path'],
                                      embed_dim=self.conf['out_dim'], 
                                      linear_layer = True if self.conf['out_dim'] != 512 else False, 
                                      n_genes=num_genes,
                                      )
            # unfreeze
            unfreeze_scgpt(gene_model.model, n_components=self.conf['scgpt_blocks'])
            
        self.print_frozen_status(gene_model)
                                
        return gene_model

    
    def get_img_model(
                    self,
                    encoder='resnet50',
                    partial_tune=False,
                    use_adapter=False,
                    adapter_bottleneck:int=None, 
                    partial_blocks=0,
                    return_tokens=False,
                    projection_head:str=None,
                    **kwargs):
        """
        Load a feature extractor model. Can currently load resnet50, UNI, CONCH, CTranspath
        Will load pretrained weights by default

        Args:
        - encoder (str): Name of the encoder to instatiate
        - mode (str): '2D' or '3D'
        - pretrained: checkpoint
        - token_set (bool): If True, 
    
        Returns:
        - model (nn.Module): Feature extractor moudule
        """
        
        if encoder == 'uni':
            model_wrapper = encoder_factory("uni")
            model = model_wrapper.model
            img_transform = model_wrapper.eval_transforms
            precision = model_wrapper.precision
        elif encoder == 'univ2':
            model_wrapper = encoder_factory("univ2")
            model = model_wrapper.model
            img_transform = model_wrapper.eval_transforms
            precision = model_wrapper.precision
        elif encoder == 'hoptimus1':
            model_wrapper = encoder_factory("hoptimus1")
            model = model_wrapper.model
            img_transform = model_wrapper.eval_transforms
            precision = model_wrapper.precision
        
        elif encoder == 'gigapath': 
            model_wrapper = encoder_factory("gigapath")
            model = model_wrapper.model
            img_transform = model_wrapper.eval_transforms
            precision = model_wrapper.precision
        
        elif encoder == 'h0mini':
            model_wrapper = encoder_factory("h0mini")
            model = model_wrapper
            img_transform = model_wrapper.eval_transforms
            precision = model_wrapper.precision
        
        elif encoder in ['virchow', 'virchow2', 'musk', 'gigapath', 'phikon2']:
            model_wrapper = encoder_factory(encoder)
            model = model_wrapper
            img_transform = model_wrapper.eval_transforms
            precision = model_wrapper.precision
    
        
        elif encoder == 'conch':
            model_wrapper = encoder_factory("conch", hf_token=kwargs.get("hf_token"))
            model = model_wrapper.model.visual
            img_transform = model_wrapper.eval_transforms
            
            # don't do resizing to 448 px for more efficient cache
            img_transform = Compose(img_transform.transforms[2:])
            precision = model_wrapper.precision

            model.proj_contrast.requires_grad_(False) 
            model.attn_pool_caption.requires_grad_(False)
            model.ln_caption.requires_grad_(False)

            if return_tokens:
                model.forward_features = model.trunk
                model.forward = types.MethodType(self.forward_tokens_2D, model)                
            else:
                model.forward = model.forward_no_head
                model.forward = types.MethodType(self.forward_conch_feats_2D, model)
        
        elif encoder == 'omiclip':
            img_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            import torch
            precision = torch.float32
            model = None

            # I don't like it, but this will have to do
            return model, img_transform, precision

        else:
            raise NotImplementedError(f"Not implemented for {encoder}!")

        if self.conf['model'] == 'linprobe' and not use_adapter: 
            # frozen params for linear probe
            for k, v in model.named_parameters():
                v.requires_grad = False
        elif partial_blocks > 0 and not use_adapter:
            model = self.unfreeze_model(model, encoder, n_components=partial_blocks)
        elif use_adapter: 
            model = VisionAdapter(model, bottleneck=adapter_bottleneck, dims=self.emb_dict[self.conf['encoder']])
        
        
        if self.conf['use_lora'] and not self.conf['model'] == 'linprobe':
            # don't wrap for linear probe 
            model = self.wrap_lora(model, partial_blocks)
        
        if 'organ_token' in self.conf and self.conf['organ_token']: 
            organ_token = True
        else: 
            organ_token=False
        
        use_img_decoder = True if self.conf['lambda_recon_img'] > 0.0 else False
        
        model = PatchRecEncoder(model, out_dim=self.conf['out_dim'], 
                                rec_dim=self.conf['n_train_genes'], 
                                add_decoder=use_img_decoder,
                                eval_transforms=img_transform, 
                                precision=precision,
                                organ_token=organ_token,
                                projection_head=projection_head,
                                dec_batch_norm=self.conf['dec_batch_norm'],
                                dec_dropout=self.conf['dec_dropout'], 
                                use_adapter=use_adapter, 
                                adapter_bottleneck=adapter_bottleneck
                                )
                
        return model, img_transform, precision

    def wrap_lora(self, model: nn.Module, partial_blocks: int): 
        from peft import LoraConfig, get_peft_model
        
        # make sure that all layers are frozen
        for k, v in model.named_parameters():
            v.requires_grad = False
        
        # get lora targets
        target_modules = self.get_lora_targets(model, partial_blocks)
        
        lora_config = LoraConfig(
            target_modules=target_modules, 
            task_type=None,
            **self.conf['lora'], 
        )
        
        print("Using LoRA with targets: ", target_modules)
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model
        


    def print_frozen_status(self, model): 
        # get trainable parameters
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Total parameters: {total_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        

    def unfreeze_model(self, model: nn.Module, encoder: str, n_components:int=0, ret_layer_list: bool=False):
        """
        Partially unfreeze the encoder.

        Args:
        - encoder (str): Encoder name
        - n_components (int): Number of blocks to unfreeze (from the output)
        """

        if n_components < 0:    # Unfreeze every parameter
            for k, v in model.named_parameters():
                v.requires_grad = True
        else: # freeze all layers otherwise (to then unfreeze the last n layers)
            for k, v in model.named_parameters():
                v.requires_grad = False
            print(encoder)
            if encoder == 'conch': # unfreeze specified trunk blocks
                assert n_components < 12
                num_list = [ 11 - idx for idx in range(n_components) ]
                incl_list = [f'trunk.blocks.{num}' for num in num_list] 
                incl_list.extend(['attn_pool_contrast', 'ln_contrast', 'head', 'trunk.norm'])
                print(incl_list)
                if ret_layer_list: 
                    return incl_list
                print("Trainable layers: ", incl_list)
                for k, v in model.named_parameters():
                    for name in incl_list:
                        if name in k:
                            v.requires_grad = True
                            break

            elif encoder == 'uni': 
                assert n_components < 24
                num_list = [ 23 - idx for idx in range(n_components)]
                incl_list = [f'blocks.{num}' for num in num_list]
                # incl_list.extend(['norm', 'fc_norm', 'head'])
                incl_list.extend(['norm.weight', 'norm.bias'])
                if ret_layer_list: 
                    return incl_list
                for k, v in model.named_parameters():
                    for name in incl_list:
                        if name in k:
                            v.requires_grad = True
                            break
            elif encoder == 'phikon2': 
                num_list = [23-idx for idx in range(n_components)]
                incl_list = [f'encoder.layer.{num}' for num in num_list]
                incl_list.extend(['layernorm.weight', 'layernorm.bias'])
                if ret_layer_list: 
                    return incl_list
                for k, v in model.named_parameters(): 
                    for name in incl_list: 
                        if name in k: 
                            v.requires_grad = True
                            break
                        
            elif encoder == 'musk': 
                pass
            elif encoder == 'univ2': 
                assert n_components < 24
                num_list = [ 23 - idx for idx in range(n_components)]
                incl_list = [f'blocks.{num}' for num in num_list]
                incl_list.extend(['norm.weight', 'norm.bias'])
                if ret_layer_list: 
                    return incl_list
                print("Trainable layers: ", incl_list)
                for k, v in model.named_parameters():
                    for name in incl_list:
                        if name in k:
                            v.requires_grad = True
                            break

            elif encoder in ['virchow', 'virchow2']: 
                assert n_components < 32
                num_list = [ 31 - idx for idx in range(n_components)]
                incl_list = [f'blocks.{num}' for num in num_list]
                incl_list.extend(['norm.weight', 'norm.bias']) 
                incl_list = ['model.'+l for l in incl_list]
                print("Trainable layers: ", incl_list)            
                for k, v in model.named_parameters():
                    for name in incl_list:
                        if name in k:
                            v.requires_grad = True
                            break
            elif encoder == 'gigapath': 
                assert n_components < 40
                num_list = [ 39 - idx for idx in range(n_components)]
                incl_list = [f'blocks.{num}' for num in num_list]
                incl_list.extend(['norm.weight', 'norm.bias']) 
                incl_list = ['model.'+l for l in incl_list]
                print("Trainable layers: ", incl_list)            
                for k, v in model.named_parameters():
                    for name in incl_list:
                        if name in k:
                            v.requires_grad = True
                            break
            elif encoder == 'h0mini': 
                assert n_components < 12
                num_list = [ 11 - idx for idx in range(n_components) ]
                incl_list = [f'blocks.{num}' for num in num_list] 
                incl_list.extend(['head', 'norm.weight', 'norm.bias'])
                incl_list = ['model.'+l for l in incl_list]
                print(incl_list)
                if ret_layer_list: 
                    return incl_list
                print("Trainable layers: ", incl_list)
                for k, v in model.named_parameters():
                    for name in incl_list:
                        if name in k:
                            v.requires_grad = True
                            break
                  
            elif encoder in ['hoptimus1', 'gigapath']: 
                assert n_components < 40
                num_list = [ 39 - idx for idx in range(n_components)]
                incl_list = [f'blocks.{num}' for num in num_list]
                incl_list.extend(['norm.weight', 'norm.bias'])
                if ret_layer_list: 
                    return incl_list
                print("Trainable layers: ", incl_list)
                for k, v in model.named_parameters():
                    for name in incl_list:
                        if name in k:
                            v.requires_grad = True
                            break

            elif encoder == 'vit_small' or encoder == 'vit_base':
                assert n_components < 12
                num_list = [ 11 - idx for idx in range(n_components) ]
                incl_list = [f'blocks.{num}' for num in num_list]
                incl_list.extend(['norm.weight', 'norm.bias']) 
                if ret_layer_list: 
                    return incl_list
                for k, v in model.named_parameters():
                    for name in incl_list:
                        if name in k:
                            v.requires_grad = True
                            break
                        
            # Gene encoders
            elif encoder == 'scgpt':
                assert n_components < 12
                num_list = [ 11 - idx for idx in range(n_components)]
                incl_list = [f'transformer_encoder.layers.{num}' for num in num_list] 
                for k, v in model.named_parameters():
                    for name in incl_list:
                        if name in k:
                            v.requires_grad = True
                            break
                
            else:
                raise NotImplementedError(f"Not implemented for {encoder}!")

        return model
    
    
    def get_lora_targets(self, model: nn.Module, partial_blocks: int):
        target_modules = set()

        head_modules = ['attn_pool_contrast', 'attn_pool_caption'] # not used in forward_no_head pass
        supported_types = (
            nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding,
            # nn.MultiheadAttention
        )
        
        encoder_patterns = {
            'conch': r"trunk\.blocks\.(\d+)", 
            'uni': r"blocks\.(\d+)",
            'univ2': r"blocks\.(\d+)",
            'phikon2': r"encoder\.layer\.(\d+)",
            'virchow': r"model\.blocks\.(\d+)",
            'virchow2': r"model\.blocks\.(\d+)",
            'h0mini': r"model\.blocks\.(\d+)",
            'hoptimus1': r"blocks\.(\d+)",
            'gigapath': r"blocks\.(\d+)",
            'musk': r"blocks\.(\d+)",
        }
        
        pattern = encoder_patterns[self.conf['encoder']]

        # Determine transformer block indices by scanning module names for the "trunk.blocks.<num>" pattern.
        block_nums = []
        for name, module in model.named_modules():
            m = re.search(pattern, name)
            if m:
                block_nums.append(int(m.group(1)))
        # breakpoint()
        max_block = max(block_nums) if block_nums else -1

        # If partial_blocks is 0, we target only the last block.
        if partial_blocks == 0:
            partial_blocks = 1
        min_block = max_block - partial_blocks + 1

        # Iterate over all modules in the model
        for name, module in model.named_modules():
            # Consider only modules of type nn.Linear
            if isinstance(module, supported_types):
                # If the module is inside a transformer block, check its block number.
                m = re.search(pattern, name)
                if m:
                    block_num = int(m.group(1))
                    # Only add modules from the last partial_blocks blocks.
                    if block_num < min_block:
                        continue
                
                
                if ("attn" in name or "mlp" in name) and not any(name.startswith(head) for head in head_modules): 
                # if ("attn" in name or "mlp" in name): 
                    target_modules.add(name)
        return sorted(list(target_modules))
        
        
        
        
        
    
    
class PatchRecEncoder(nn.Module): 
    """
    Wrapper function for vision encoder that uses the encoder as default but provides access to 
    the decoder
    """
        
    def __init__(self, encoder, out_dim, rec_dim, eval_transforms, precision, 
                 organ_token:bool=False, 
                 add_decoder:bool=True, 
                 projection_head:str=None,
                 dec_batch_norm:bool=False,
                 dec_dropout:float=0.0, 
                 use_adapter:bool=False, 
                 adapter_bottleneck: int=256, 
                 ): 
        super().__init__()
        self.encoder = encoder
        self.organ_token = organ_token
        self.use_adapter = use_adapter
        
        if self.use_adapter: 
            self.adapter = self.encoder.adapter
        
        organ_ids = find_organ_ids()
        num_organs = len(organ_ids)
        
        if self.organ_token:
            self.organ_emb = nn.Embedding(num_organs, out_dim)
        
        if projection_head == 'linear': 
            self.projection_head = nn.Linear(out_dim, out_dim)
        elif projection_head == 'mlp':
            self.projection_head = create_mlp(
                in_dim=out_dim, 
                out_dim=out_dim, 
                batch_norm=False
            )
        else: 
            self.projection_head = None        
        
        self.is_lora = True if isinstance(self.encoder, PeftModel) else False
        if add_decoder:
            self.decoder = ImgEmbDecoder(out_dim, rec_dim, batch_norm=dec_batch_norm, dropout=dec_dropout)
        else: 
            self.decoder = None
        
        self.eval_transforms = eval_transforms
        self.precision = precision
         
    def forward(self, x, decode: bool=False, organ_id=None, projection: bool=False):
        z = self.encoder(x)
        if self.organ_token and organ_id is not None: 
            z = z + self.organ_emb(organ_id)
        
        if self.use_adapter: 
            z = self.adapter(z)
        
        if decode and self.decoder is not None: 
            y = self.decoder(z)
        
        if self.projection_head is not None: 
            z_proj = self.projection_head(z)
        if projection: 
            z = z_proj if self.projection_head is not None else z
        
        
        if decode and self.decoder is not None:
            return z, y
        else: 
            return z

class ImgEmbDecoder(nn.Module): 
    def __init__(self, out_dim, rec_dim, hid_dim:int=512, batch_norm:bool=False, dropout:float=0.0): 
        super().__init__()
        layers = []
        layers.append(nn.Linear(out_dim, hid_dim))
        if batch_norm: 
            layers.append(nn.BatchNorm1d(hid_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hid_dim, rec_dim))
        layers.append(nn.ReLU())
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.decoder(x)
        
        
class PathomClipWrapper(nn.Module): 
    
    def __init__(self, encoder: nn.Module, out_dim: int, n_train_genes: int): 
        super().__init__()
        self.encoder = encoder
        self.decoder = self.encoder.decoder
        self.local_transformer = nn.TransformerEncoderLayer(
            d_model=out_dim, 
            nhead=8, 
            dim_feedforward=2048, 
            dropout=0.1
        )
                
    def forward(self, x, decode: bool=False, **kwargs): 
        z = self.encoder(x)
        z = self.local_transformer(z)
        if decode:
            y = self.decoder(z)
            return z, y
        else: 
            return z
