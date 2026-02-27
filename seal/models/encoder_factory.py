from __future__ import annotations
from torchvision import transforms
import traceback
import argparse
import timm
from typing import Dict, List, Optional
import torch.nn as nn
from pathlib import Path
from abc import abstractmethod
from seal.utils.exp_utils import update_config
from seal.utils.constants import EMB_DICT
import torch
import subprocess
import shutil
import os

from pathlib import Path
import argparse

def find_config_yaml():
    # 1. Check for local override first (preserves original behavior)
    local_config = Path("conf/config.yaml")
    if local_config.exists():
        return str(local_config)
    
    # 2. Fall back to the bundled package config
    package_config = Path(__file__).resolve().parent.parent / "conf" / "config.yaml"
    if package_config.exists():
        return str(package_config)
    
    # 3. Last resort fallback to avoid a silent failure
    raise FileNotFoundError("Could not find config.yaml locally or in the installed package.")

def get_constants(norm='imagenet'):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
    OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]
    NONE_MEAN = None
    NONE_STD = None
    if norm == 'imagenet':
        return IMAGENET_MEAN, IMAGENET_STD
    elif norm == 'openai_clip':
        return OPENAI_MEAN, OPENAI_STD
    elif norm == 'none':
        return NONE_MEAN, NONE_STD
    else:
        raise ValueError(f"Invalid norm: {norm}")


def get_eval_transforms(mean, std, target_img_size = -1, center_crop = False, **resize_kwargs):
    trsforms = []
    
    if target_img_size > 0:
        trsforms.append(transforms.Resize(target_img_size, **resize_kwargs))
    if center_crop:
        assert target_img_size > 0, "target_img_size must be set if center_crop is True"
        trsforms.append(transforms.CenterCrop(target_img_size))
        
    
    trsforms.append(transforms.ToTensor())
    if mean is not None and std is not None:
        trsforms.append(transforms.Normalize(mean, std))
    trsforms = transforms.Compose(trsforms)

    return trsforms


class InferenceEncoder(torch.nn.Module):
    
    def __init__(self, weights_path=None, **build_kwargs):
        super(InferenceEncoder, self).__init__()
        
        self.weights_path = weights_path
        self.model, self.eval_transforms, self.precision = self._build(weights_path, **build_kwargs)
        
    def forward(self, x):
        z = self.model(x)
        return z
        
    @abstractmethod
    def _build(self, **build_kwargs):
        pass
        

class ConchInferenceEncoder(InferenceEncoder):
    def _build(self, _, hf_token: Optional[str] = None):
        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except Exception:
            traceback.print_exc()
            raise Exception("Please install CONCH `pip install git+https://github.com/Mahmoodlab/CONCH.git`")

        try:
            model, preprocess = create_model_from_pretrained(
                'conch_ViT-B-16',
                "hf_hub:MahmoodLab/conch",
                hf_auth_token=hf_token,
            )
        except TypeError:
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
            model, preprocess = create_model_from_pretrained(
                'conch_ViT-B-16',
                "hf_hub:MahmoodLab/conch",
            )
        except Exception:
            traceback.print_exc()
            raise Exception("Failed to download CONCH model, make sure that you were granted access and that you correctly registered your token")
        
        eval_transform = preprocess
        precision = torch.float32
        
        return model, eval_transform, precision
    
    def forward(self, x):
        return self.model.encode_image(x, proj_contrast=False, normalize=False)
    


    
class CustomInferenceEncoder(InferenceEncoder):
    def __init__(self, weights_path, name, model, transforms, precision):
        super().__init__(weights_path)
        self.model = model
        self.transforms = transforms
        self.precision = precision
        
    def _build(self, weights_path):
        return None, None, None
    

class PhikonInferenceEncoder(InferenceEncoder):

    def _load(self):
        from transformers import ViTModel
        model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        return model

    def _build(self, _, return_cls=True):
        self.return_cls = return_cls
        model = self._load()
        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std)
        precision = torch.float32
        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.forward_features(x)
        if self.return_cls:
            out = out.last_hidden_state[:, 0, :]
        else:
            class_token = out.last_hidden_state[:, 0, :]
            patch_tokens = out.last_hidden_state[:, 1:, :]
            out = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        return out
    
    def forward_features(self, x):
        out = self.model(pixel_values=x)
        return out
    

# class ViTBaseInferenceEncoder(InferenceEncoder):


class PhikonV2InferenceEncoder(PhikonInferenceEncoder):
    def _load(self):
        from transformers import AutoModel
        model = AutoModel.from_pretrained("owkin/phikon-v2")
        return model
    
    
    
class ResNet50InferenceEncoder(InferenceEncoder):
    def _build(
        self, 
        _,
        pretrained=True, 
        timm_kwargs={"features_only": True, "out_indices": [3], "num_classes": 0},
        pool=True
    ):
        import timm

        model = timm.create_model("resnet50.tv_in1k", pretrained=pretrained, **timm_kwargs)
        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std)
        precision = torch.float32
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
        
        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.forward_features(x)
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out
    
    def forward_features(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        return out
                     
    
class UNIInferenceEncoder(InferenceEncoder):
    def _build(
        self, 
        _,
        timm_kwargs={"dynamic_img_size": True, "num_classes": 0, "init_values": 1.0}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        
        try:
            model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, **timm_kwargs)
        except:
            traceback.print_exc()
            raise Exception("Failed to download UNI model, make sure that you were granted access and that you correctly registered your token")
        
        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        precision = torch.float16
        return model, eval_transform, precision
    
    
class UNIv2InferenceEncoder(InferenceEncoder):

    def _build(self, _, 
               timm_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
               ):

        self.enc_name = 'uni_v2'
        
        try:
            model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        except:
            traceback.print_exc()
            raise Exception("Failed to download UNI v2 model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        precision = torch.bfloat16
        return model, eval_transform, precision
    
class HOptimus1InferenceEncoder(InferenceEncoder):

    def _build(
        self, _,
        timm_kwargs={'init_values': 1e-5, 'dynamic_img_size': False},
    ):
        import timm
        assert timm.__version__ == '0.9.16', f"H-Optimus requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
        from torchvision import transforms

        self.enc_name = 'hoptimus1'
        try:
            model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True, **timm_kwargs)
        except:
            traceback.print_exc()
            raise Exception("Failed to download HOptimus-1 model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = transforms.Compose([
            transforms.Resize(224),  
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
        
        precision = torch.float16
        return model, eval_transform, precision


class GigaPathInferenceEncoder(InferenceEncoder):
    def _build(
        self, 
        _,
        timm_kwargs={}
        ):
        import timm
        from torchvision import transforms
        
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, **timm_kwargs)

        eval_transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        precision = torch.float32
        return model, eval_transform, precision



class H0MiniInferenceEncoder(InferenceEncoder):
    def _build(
        self, 
        _,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
        ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        
        model = timm.create_model("hf-hub:bioptimus/H0-mini", pretrained=True, **timm_kwargs)

        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

        precision = torch.float32
        return model, eval_transform, precision
    
    def forward(self, x):
        output = self.model(x)
        
        cls_features = output[:, 0]
        
        patch_token_features = output[:, self.model.num_prefix_tokens :]
        
        concatenated_features = torch.cat(
            [cls_features, patch_token_features.mean(1)], dim=-1
        )
        
        return concatenated_features.to(self.precision)
    

    
    
class VirchowInferenceEncoder(InferenceEncoder):
    import timm
    
    def _build(
        self,
        _,
        return_cls=False,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            **timm_kwargs
        )
        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

        precision = torch.float32
        self.return_cls = return_cls
        
        return model, eval_transform, precision
        
    def forward(self, x):
        output = self.model(x)
        class_token = output[:, 0]

        if self.return_cls:
            return class_token
        else:
            patch_tokens = output[:, 1:]
            embeddings = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            return embeddings
        

class Virchow2InferenceEncoder(InferenceEncoder):
    import timm
    
    def _build(
        self,
        _,
        return_cls=False,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            **timm_kwargs
        )
        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        precision = torch.float16
        self.return_cls = return_cls
        
        return model, eval_transform, precision

    def forward(self, x):
        output = self.model(x)
    
        if self.return_cls:
            return class_token
        
        class_token = output[:, 0]
        patch_tokens = output[:, 5:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        return embedding

class HOptimus0InferenceEncoder(InferenceEncoder):
    
    def _build(
        self,
        _,
        timm_kwargs={'init_values': 1e-5, 'dynamic_img_size': False}
    ):
        import timm
        from torchvision import transforms

        model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, **timm_kwargs)

        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
        
        precision = torch.float16
        return model, eval_transform, precision
         
    
class MuskInferenceEncoder(InferenceEncoder):
    
    # timm_kwargs = {'inference_aug': False, 'with_proj': False, 'out_norm': False, 'return_global': True}
    
    def _build(self, 
               _, 
               timm_kwargs = {'inference_aug': False, 'with_proj': False, 'out_norm': False, 'return_global': True},
               ):
        '''
        Args:
            inference_aug (bool): Whether to use test-time multiscale augmentation. Default is False to allow for fair comparison with other models.
        '''
        self.enc_name = 'musk'
        # self.inference_aug = inference_aug
        # self.with_proj = with_proj
        # self.out_norm = out_norm
        # self.return_global = return_global
    
        try:
            from musk import utils, modeling
        except:
            traceback.print_exc()
            raise Exception("Please install MUSK `pip install git+https://github.com/lilab-stanford/MUSK`")
        
        try:
            from timm.models import create_model 
            model = create_model("musk_large_patch16_384")
            utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')
        except:
            traceback.print_exc()
            raise Exception("Failed to download MUSK model, make sure that you were granted access and that you correctly registered your token")
        
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        from torchvision.transforms import Compose, Resize, InterpolationMode
        eval_transform = get_eval_transforms(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, target_img_size = 384, center_crop = True, interpolation=InterpolationMode.BICUBIC, antialias=True)
        precision = torch.float16
        
        return model, eval_transform, precision
    
    def forward(self, x):
        return self.model(
                image=x,
                with_head=self.with_proj,
                out_norm=self.out_norm,
                ms_aug=self.inference_aug,
                return_global=self.return_global  
                )[0]  # Forward pass yields (vision_cls, text_cls). We only need vision_cls.
    
    
def encoder_factory(enc_name, **build_kwargs):
    if enc_name == 'conch':
        return ConchInferenceEncoder(**build_kwargs)
    elif enc_name == 'uni':
        return UNIInferenceEncoder()
    elif enc_name == 'univ2': 
        return UNIv2InferenceEncoder()
    elif enc_name == 'phikon':
        return PhikonInferenceEncoder()
    elif enc_name == 'resnet50':
        return ResNet50InferenceEncoder()
    elif enc_name == 'phikon2': 
        return PhikonV2InferenceEncoder()
    elif enc_name == 'gigapath':
        return GigaPathInferenceEncoder()
    elif enc_name == 'virchow':
        return VirchowInferenceEncoder()
    elif enc_name == 'virchow2':
        return Virchow2InferenceEncoder()
    elif enc_name == 'h0mini':
        return H0MiniInferenceEncoder()
    elif enc_name == 'hoptimus0':
        return HOptimus0InferenceEncoder()
    elif enc_name == 'hoptimus1':
        return HOptimus1InferenceEncoder()
    elif enc_name == 'musk':
        return MuskInferenceEncoder()
    else:
        raise ValueError(f"Unknown encoder name {enc_name}")
    

def _checkpoint_name_candidates(backbone: str, component: str) -> List[str]:
    if component == "vision":
        return [f"seal_{backbone}_vision.pth", f"{backbone}-seal-vision.pth"]
    if component == "omics":
        return [f"seal_{backbone}_omics.pth", f"{backbone}-seal-omics.pth"]
    raise ValueError(f"Unsupported checkpoint component: {component}")


def _resolve_local_checkpoints(backbone: str) -> Dict[str, Optional[Path]]:
    checkpoint_dir = Path(f"weights/{backbone}_SEAL/")
    resolved: Dict[str, Optional[Path]] = {"vision": None, "omics": None}
    for component in ("vision", "omics"):
        for filename in _checkpoint_name_candidates(backbone, component):
            candidate = checkpoint_dir / filename
            if candidate.exists():
                resolved[component] = candidate
                break
    return resolved


def _run_hf_cli_download(
    repo_id: str,
    filename: str,
    local_dir: str,
    revision: Optional[str] = "main",
    token: Optional[str] = None,
    timeout_s: int = 300,
) -> Path:
    hf_bin = shutil.which("hf")
    hf_cli_bin = shutil.which("huggingface-cli")
    if hf_bin:
        cmd = [
            hf_bin,
            "download",
            repo_id,
            filename,
            "--revision",
            revision or "main",
            "--local-dir",
            local_dir,
        ]
    elif hf_cli_bin:
        cmd = [
            hf_cli_bin,
            "download",
            repo_id,
            filename,
            "--revision",
            revision or "main",
            "--local-dir",
            local_dir,
        ]
    else:
        raise RuntimeError(
            "Hugging Face CLI not found. Install with `pip install -U \"huggingface_hub[cli]\"` "
            "and ensure `hf` or `huggingface-cli` is on PATH."
        )

    env = os.environ.copy()
    if token:
        env["HF_TOKEN"] = token
        cmd.extend(["--token", token])

    print(f"[SEAL] Downloading {filename} from {repo_id} via Hugging Face CLI...")
    result = subprocess.run(
        cmd,
        env=env,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"HF CLI download failed for {repo_id}/{filename} (exit={result.returncode}). "
            f"Command: {' '.join(cmd)}"
        )

    downloaded_path = Path(local_dir) / filename
    if not downloaded_path.exists():
        raise RuntimeError(
            f"HF CLI reported success but file not found at expected path: {downloaded_path}"
        )
    return downloaded_path


def _resolve_hf_checkpoints(
    backbone: str,
    hf_repo_id: str,
    hf_revision: Optional[str] = "main",
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> tuple[Dict[str, Optional[Path]], Dict[str, List[str]]]:
    resolved: Dict[str, Optional[Path]] = {"vision": None, "omics": None}
    errors: Dict[str, List[str]] = {"vision": [], "omics": []}
    local_dir = str(Path(f"weights/{backbone}_SEAL/"))
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    for component in ("vision", "omics"):
        for candidate in _checkpoint_name_candidates(backbone, component):
            try:
                downloaded = _run_hf_cli_download(
                    repo_id=hf_repo_id,
                    filename=candidate,
                    local_dir=local_dir,
                    revision=hf_revision,
                    token=hf_token,
                )
                resolved[component] = downloaded
                break
            except Exception as exc:
                errors[component].append(f"{candidate}: {exc}")
                continue

    return resolved, errors


def _resolve_hf_token(hf_token: Optional[str]) -> Optional[str]:
    return (
        hf_token
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_API_KEY")
    )


def _hf_cli_available() -> bool:
    return shutil.which("hf") is not None or shutil.which("huggingface-cli") is not None


def _validate_hf_token(hf_token: str) -> None:
    if not hf_token.startswith("hf_"):
        raise RuntimeError(
            "Hugging Face token appears invalid (expected prefix 'hf_')."
        )
    

def seal_factory(
    backbone: str,
    gene_post_warmup: bool = False,
    source: str = "auto",
    hf_repo_id: str = "MahmoodLab/SEAL",
    hf_revision: Optional[str] = "main",
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[nn.Module]:
    """Lightweight wrapper to load encoder with checkpoint
    
    Args: 
        backbone (str): Name of the backbone encoder
        gene_post_warmup (bool): Whether to load the post-warmup (pre-alignmnt) gene model checkpoint
        source (str): One of ['auto', 'hf', 'local']
        hf_repo_id (str): Hugging Face repository ID containing SEAL checkpoints
        hf_revision (str): Hugging Face revision (branch, tag, or commit)
        hf_token (str): Hugging Face token for private repos
        cache_dir (str): Optional cache directory for downloaded files
        
    ```    
    
    """
    if source not in {"auto", "hf", "local"}:
        raise ValueError(f"Invalid source '{source}'. Expected one of: auto, hf, local.")

    resolved_hf_token = _resolve_hf_token(hf_token)
    hf_cli_is_available = _hf_cli_available()
    if source == "hf" and not hf_cli_is_available:
        raise RuntimeError(
            "Hugging Face CLI not found. Install with `pip install -U \"huggingface_hub[cli]\"` "
            "and ensure `hf` or `huggingface-cli` is on PATH."
        )
    if source == "auto" and not hf_cli_is_available:
        print(
            "Warning: Hugging Face CLI not found (`hf`/`huggingface-cli`). "
            "Skipping HF download and falling back to local checkpoints."
        )

    if backbone == "conch":
        if not resolved_hf_token:
            raise RuntimeError(
                "Missing Hugging Face token for CONCH. "
                "Set HF_TOKEN (or pass hf_token=...) before calling seal_factory."
            )
        _validate_hf_token(resolved_hf_token)

    local_ckpts = _resolve_local_checkpoints(backbone)
    hf_ckpts: Dict[str, Optional[Path]] = {"vision": None, "omics": None}
    hf_errors: Dict[str, List[str]] = {"vision": [], "omics": []}

    if source in {"auto", "hf"} and hf_cli_is_available:
        hf_ckpts, hf_errors = _resolve_hf_checkpoints(
            backbone=backbone,
            hf_repo_id=hf_repo_id,
            hf_revision=hf_revision,
            hf_token=resolved_hf_token,
            cache_dir=cache_dir,
        )

    if source == "local":
        vision_ckpt = local_ckpts["vision"]
        omics_ckpt = local_ckpts["omics"]
    elif source == "hf":
        vision_ckpt = hf_ckpts["vision"]
        omics_ckpt = hf_ckpts["omics"]
        missing = []
        if vision_ckpt is None:
            missing.append("vision")
        if omics_ckpt is None:
            missing.append("omics")
        if missing:
            details = []
            for component in missing:
                err_lines = hf_errors.get(component) or []
                if err_lines:
                    details.append(f"{component} attempts:\n  - " + "\n  - ".join(err_lines))
                else:
                    details.append(f"{component} attempts: no successful candidate and no captured error output")
            raise RuntimeError(
                f"Failed to download required HF checkpoints for backbone '{backbone}' "
                f"from repo '{hf_repo_id}' in source='hf' mode.\n" + "\n".join(details)
            )
    else:
        vision_ckpt = hf_ckpts["vision"] if hf_ckpts["vision"] is not None else local_ckpts["vision"]
        omics_ckpt = hf_ckpts["omics"] if hf_ckpts["omics"] is not None else local_ckpts["omics"]

    if vision_ckpt is None:
        print(
            f"Warning: No vision checkpoint found for backbone '{backbone}' "
            f"(source={source}, hf_repo_id={hf_repo_id}). Returning None image model."
        )
    if omics_ckpt is None:
        print(
            f"Warning: No omics checkpoint found for backbone '{backbone}' "
            f"(source={source}, hf_repo_id={hf_repo_id}). Returning None gene model."
        )

    ckpt_dir = Path(f"weights/{backbone}_SEAL/")
    img_encoder, img_transforms, img_precision = load_img_model_from_checkpoint(
        checkpoint_dir=ckpt_dir,
        checkpoint_path=vision_ckpt,
        hf_token=resolved_hf_token,
    )
    gene_encoder = load_gene_model_from_checkpoint(
        checkpoint_dir=ckpt_dir,
        post_warmup=gene_post_warmup,
        checkpoint_path=omics_ckpt,
    )
    
    
    return (img_encoder, img_transforms, img_precision), gene_encoder



    
def fill_missing_config_keys(model_config, default_config):
    """Fill missing keys in model_config with default_config. 
    This is for the edge case where the model_config checkpoint was written 
    before additional keys were added to the default_config. 
    """
    for key, value in default_config.items():
        if key not in model_config:
            model_config[key] = value
    return model_config
    
    
def load_img_model_from_checkpoint(
    checkpoint_dir,
    checkpoint_path: Optional[Path] = None,
    hf_token: Optional[str] = None,
): 
    import argparse
    from seal.models.load_model import ModelMixin
    
    default_config_path = argparse.Namespace(config=find_config_yaml())

    if checkpoint_path is None:
        print(f"Warning: No image checkpoint provided for checkpoint_dir={checkpoint_dir}")
        return None, None, None

    model_config = update_config(default_config_path)
    default_config = update_config(default_config_path)
    
    # fill missing keys if checkpoint was written before additional keys were added (required to load model instance)
    model_config = fill_missing_config_keys(model_config, default_config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # get model
    mixin = ModelMixin()
    mixin.conf = model_config
    mixin.emb_dict = EMB_DICT
    
    model, img_transform, precision = mixin.get_img_model(model_config['encoder'], 
                                                            partial_blocks=model_config['partial_blocks'], 
                                                            use_adapter=model_config['use_adapter'],
                                                            adapter_bottleneck=model_config['adapter_bottleneck'],
                                                            hf_token=hf_token,
                                                            )
    
    
    model.enc_name = f"{checkpoint_dir.name if checkpoint_dir is not None else Path(checkpoint_path).stem}"
    
    ckpt_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    if any(k.startswith("module.") for k in ckpt_dict.keys()): 
        # remove 'module.' prefix
        ckpt_dict = {k.replace('module.', ''): v for k, v in ckpt_dict.items()}
    
    if any('encoder.base_model.model.encoder.' in k for k in ckpt_dict.keys()):
        # virchow2 quirk
        ckpt_dict = {k.replace('encoder.base_model.model.encoder.', 'encoder.base_model.model.model.'): v for k, v in ckpt_dict.items()}
        
    missing, unexpected = model.load_state_dict(ckpt_dict, strict=False)
    
    if len(unexpected) > 0: 
        print(f"WARNING: Unexpected keys in checkpoint: {unexpected}")
    
    model.eval()
    
    return model, img_transform, precision

def load_gene_model_from_checkpoint(
    checkpoint_dir,
    post_warmup: bool = False,
    checkpoint_path: Optional[Path] = None,
): 
    """
    Load in gene model checkpoint if available
    
    Args: 
        checkpoint_dir (Path): Path to checkpoint directory
        warmup (bool): Whether to get the post-warmup (pre-alignment) gene model checkpoint
    """
    from seal.models.load_model import ModelMixin
    
    default_config_path = argparse.Namespace(config=find_config_yaml())
    model_config = update_config(default_config_path)
    default_config = update_config(default_config_path)
    
    # fill missing keys if checkpoint was written before additional keys were added (required to load model instance)
    model_config = fill_missing_config_keys(model_config, default_config)
    
    mixin = ModelMixin()
    mixin.conf = model_config
    
    gene_model = mixin.get_gene_model(num_genes=model_config['n_train_genes'])
    
    if checkpoint_path is None:
        print(f"Warning: No gene model checkpoint available for run in {checkpoint_dir}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    
    ckpt_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    if any(k.startswith("module.") for k in ckpt_dict.keys()): 
        # remove 'module.' prefix
        ckpt_dict = {k.replace('module.', ''): v for k, v in ckpt_dict.items()}
    
    missing, unexpected = gene_model.load_state_dict(ckpt_dict, strict=False)
    gene_model.eval()
    
    return gene_model
