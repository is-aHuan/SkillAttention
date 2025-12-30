import random
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


def replace_bn_with_gn(module: nn.Module, num_groups_div: int = 16):
    """
    递归把 module 里的 BatchNorm2d 替换成 GroupNorm。
    num_groups = num_features // num_groups_div
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = max(1, num_channels // num_groups_div)
            gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups_div=num_groups_div)
    return module

# Image Encoder
# From https://github.com/suraj-nair-1/lorel/blob/main/models.py
class BaseEncoder(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self):
        super().__init__()


    def preprocess(self, observations):
        """
            Reshape to 4 dimensions so it works for convolutions
            Chunk the time and batch dimensions
        """
        B, T, C, H, W = observations.shape
        return observations.reshape(-1, C, H, W).type(torch.float32).contiguous()
        
    def unpreprocess(self, embeddings, B, T):
        """
            Reshape back to 5 dimensions 
            Unsqueeze the batch and time dimensions
        """
        BT, E = embeddings.shape
        return embeddings.reshape(B, T, E)
        

# Image Encoder
# From https://github.com/suraj-nair-1/lorel/blob/main/models.py
class Encoder(BaseEncoder):
    __constants__ = ['embedding_size']

    def __init__(self, hidden_size, activation_function='relu', ch=3, robot=False):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.robot = robot
        if self.robot:
            # g = 4
            g = ch // 3
        else:
            g = 1
        self.conv1 = nn.Conv2d(ch, 32, 4, stride=2, padding=1, groups=g)  # 3
        self.conv1_2 = nn.Conv2d(32, 32, 4, stride=1, padding=1, groups=g)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1, groups=g)
        self.conv2_2 = nn.Conv2d(64, 64, 4, stride=1, padding=1, groups=g)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1, groups=g)
        self.conv3_2 = nn.Conv2d(128, 128, 4, stride=1, padding=1, groups=g)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=1, groups=g)
        self.conv4_2 = nn.Conv2d(256, 256, 4, stride=1, padding=1, groups=g)

        self.fc1 = nn.Linear(1024, 512)
        self.fc1_2 = nn.Linear(512, 512)
        self.fc1_3 = nn.Linear(512, 512)
        self.fc1_4 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, hidden_size)

    def forward(self, observations):
        if self.robot:
            observations = torch.cat([
                observations[:, :3], observations[:, 12:15], observations[:, 3:6], observations[:, 15:18],
                observations[:, 6:9], observations[:, 18:21], observations[:, 9:12], observations[:, 21:],
            ], 1)
        if len(observations.shape) == 5:
            preprocessed_observations = self.preprocess(observations)
        else:
            preprocessed_observations = observations
        hidden = self.act_fn(self.conv1(preprocessed_observations))
        hidden = self.act_fn(self.conv1_2(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv2_2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv3_2(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = self.act_fn(self.conv4_2(hidden))
        hidden = hidden.reshape(preprocessed_observations.shape[0], -1)

        hidden = self.act_fn(self.fc1(hidden))
        hidden = self.act_fn(self.fc1_2(hidden))
        hidden = self.act_fn(self.fc1_3(hidden))
        hidden = self.act_fn(self.fc1_4(hidden))
        hidden = self.fc2(hidden)
        
        if len(observations.shape) == 5:
            return self.unpreprocess(hidden, observations.shape[0], observations.shape[1])
        else:
            return hidden

class ProprioMLPEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int = 128, 
                 hidden_dims: list = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            # 使用 Mish 或 ReLU，Diffusion 中 Mish 也很常用，这里用 ReLU
            layers.append(nn.ReLU()) 
            prev_dim = h_dim
            
        # 最终输出层 (无激活函数)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        self._output_dim = output_dim

    @property
    def feat_dim(self):
        return self._output_dim

    def forward(self, x):
        """
        x: [B*T, Dp]
        output: [B*T, 128]
        """
        return self.net(x)

class ResNetImageEncoder(nn.Module):
    """
    单路 RGB 图像 encoder（无内部 resize）：
      - 可选 random / center crop 到 crop_shape
      - ImageNet norm（可选）
      - ResNet18 backbone（去掉 fc，可选 BN→GN）

    输入:
      x: [B, 3, H, W]，H,W 由数据集保证为 240x320 或其它固定大小

    输出:
      feats: [B, feat_dim]，resnet18 默认 feat_dim = 512
    """
    def __init__(self,
                 crop_shape=(216, 288),
                 random_crop=True,
                 use_group_norm=True,
                 imagenet_norm=False,
                 resnet_name='resnet18',
                 pretrained=False):
        super().__init__()
        self.crop_shape = crop_shape      # (Hc, Wc) or None
        self.random_crop = random_crop
        self.imagenet_norm = imagenet_norm

        # 1) 构造 resnet backbone
        if resnet_name == 'resnet18':
            resnet = models.resnet18(
                weights=None if not pretrained else models.ResNet18_Weights.IMAGENET1K_V1
            )
        else:
            raise ValueError(f"Unsupported resnet backbone: {resnet_name}")

        # 可选 BN -> GN
        if use_group_norm:
            resnet = replace_bn_with_gn(resnet, num_groups_div=16)

        # 去掉最后的 fc，保留到 avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # conv..layer4 + avgpool
        self.feat_dim = resnet.fc.in_features  # resnet18 = 512

        # ImageNet 标准化参数
        if imagenet_norm:
            self.register_buffer(
                'img_mean',
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                'img_std',
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )
        else:
            self.img_mean = None
            self.img_std = None

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W]，裁剪成 [B,3,Hc,Wc]
        """
        if self.crop_shape is None:
            return x

        _, _, H, W = x.shape
        Hc, Wc = self.crop_shape
        Hc = min(Hc, H)
        Wc = min(Wc, W)

        if self.random_crop and self.training:
            top = random.randint(0, H - Hc)
            left = random.randint(0, W - Wc)
        else:
            # eval / rollout 用中心裁剪，保证稳定
            top = (H - Hc) // 2
            left = (W - Wc) // 2

        return x[:, :, top:top + Hc, left:left + Wc]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W]
        """
        # 1) crop（H,W 由数据集保证，比如 240x320）
        x = self._crop(x)

        # 2) ImageNet norm
        if self.imagenet_norm and self.img_mean is not None:
            x = (x - self.img_mean) / self.img_std

        # 3) ResNet18 backbone
        feats = self.backbone(x)      # [B,512,1,1]（全局 ave pool 完了）
        feats = feats.view(feats.size(0), -1)  # [B,512]
        return feats

class SingleImageObsEncoder(nn.Module):
    """
    单路图像版 ObsEncoder（对齐 MultiObsEncoder 风格）

    输入:
      image:   [B*T, 3, H, W]
      proprio: [B*T, Dp]  (可选)

    输出:
      obs_embed:
        - 无 proprio: [B*T, 512]
        - 有 proprio: [B*T, 512 + Dp]
    """
    def __init__(self,
                 proprio_input_dim: int = 0,
                 crop_shape=(64, 64),
                 random_crop: bool = True,
                 use_group_norm: bool = True,
                 imagenet_norm: bool = True,
                 resnet_name: str = 'resnet18',
                 pretrained: bool = False):
        super().__init__()

        # -------- 图像 encoder --------
        self.image_encoder = ResNetImageEncoder(
            crop_shape=crop_shape,
            random_crop=random_crop,
            use_group_norm=use_group_norm,
            imagenet_norm=imagenet_norm,
            resnet_name=resnet_name,
            pretrained=pretrained
        )

        # -------- proprio encoder（可选） --------
        if proprio_input_dim > 0:
            self.proprio_encoder = ProprioMLPEncoder(
                input_dim=proprio_input_dim,
                output_dim=proprio_input_dim,
                hidden_dims=[128, 64]
            )
        else:
            self.proprio_encoder = None

    # ---------- feature dim ----------
    @property
    def image_feat_dim(self):
        return self.image_encoder.feat_dim  # e.g. 512

    @property
    def proprio_feat_dim(self):
        if self.proprio_encoder is None:
            return 0
        return self.proprio_encoder.feat_dim

    @property
    def feat_dim(self):
        return self.image_feat_dim + self.proprio_feat_dim

    # ---------- forward ----------
    def forward(self, image, proprio=None):
        """
        image:   [B*T, 3, H, W]
        proprio: [B*T, Dp] or None
        """
        # 图像特征
        img_feat = self.image_encoder(image)  # [B*T, 512]

        if self.proprio_encoder is None:
            return img_feat

        assert proprio is not None, "proprio_input_dim > 0 but proprio is None"
        proprio_feat = self.proprio_encoder(proprio)  # [B*T, Dp]

        obs_embed = torch.cat([img_feat, proprio_feat], dim=-1)
        return obs_embed

class MultiObsEncoder(nn.Module):
    """
    ResNet18 版 MultiObsEncoder：
      - head / wrist 各一路 ResNetImageEncoder
      - proprio 直接 concat

    输入:
      head_image:  [B*T, 3, H, W]
      wrist_image: [B*T, 3, H, W]
      proprio:     [B*T, Dp]

    输出:
      obs_embed:   [B*T, 512 + 512 + Dp]
    """
    def __init__(self,
                 proprio_input_dim: int,
                 share_rgb_model: bool = False,
                 crop_shape=(216, 288),
                 random_crop: bool = True,
                 use_group_norm: bool = True,
                 imagenet_norm: bool = True,
                 resnet_name: str = 'resnet18',
                 pretrained: bool = False,
                 **kargs):
        super().__init__()

        # head 视角 encoder
        self.head_encoder = ResNetImageEncoder(
            crop_shape=crop_shape,
            random_crop=random_crop,
            use_group_norm=use_group_norm,
            imagenet_norm=imagenet_norm,
            resnet_name=resnet_name,
            pretrained=pretrained
        )

        # wrist 视角 encoder
        if share_rgb_model:
            self.wrist_encoder = self.head_encoder
        else:
            self.wrist_encoder = ResNetImageEncoder(
                crop_shape=crop_shape,
                random_crop=random_crop,
                use_group_norm=use_group_norm,
                imagenet_norm=imagenet_norm,
                resnet_name=resnet_name,
                pretrained=pretrained
            )

        self.proprio_encoder = ProprioMLPEncoder(
            input_dim=proprio_input_dim,
            output_dim=proprio_input_dim,
            hidden_dims=[128, 64] # 两层隐藏层，增加非线性表达能力
        )

    @property
    def head_feat_dim(self):
        return self.head_encoder.feat_dim

    @property
    def wrist_feat_dim(self):
        return self.wrist_encoder.feat_dim
    
    @property
    def proprio_feat_dim(self):
        return self.proprio_encoder.feat_dim

    def forward(self, head_image, wrist_image, proprio):
        """
        head_image:  [B*T, 3, H, W]
        wrist_image: [B*T, 3, H, W]
        proprio:     [B*T, Dp]
        """
        assert head_image.shape == wrist_image.shape

        # 单路编码: [B*T, 512], [B*T, 512]
        head_feat  = self.head_encoder(head_image)
        wrist_feat = self.wrist_encoder(wrist_image)

        # MLP 提供了一个可学习的投影，比直接 concat 更好
        proprio_feat = self.proprio_encoder(proprio)

        # proprio: [B*T,Dp]，前面 Dataset 已经 normalize 过
        obs_embed = torch.cat([head_feat, wrist_feat, proprio_feat], dim=-1)
        # 形状: [B*T, 512+512+Dp]
        return obs_embed


    