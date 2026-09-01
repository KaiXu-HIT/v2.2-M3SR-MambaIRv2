import torch
from torch import nn

from basicsr.archs.rgb_depth_mambairv2_arch import RGBDepthMambaIRv2
from basicsr.utils.registry import ARCH_REGISTRY

try:
    from transformers import AutoTokenizer, CLIPTextModel
except ImportError:
    AutoTokenizer = None
    CLIPTextModel = None


@ARCH_REGISTRY.register()
class RGBDepthTextMambaIRv2(RGBDepthMambaIRv2):
    """Stage-1 B3: RGB-depth early fusion followed by global text FiLM.

    B3 change note: the B1 RGB/depth convolutions and 1x1 fusion are retained.
    A frozen CLIP sentence encoder produces gamma/beta values that modulate the
    fused feature exactly once before the unchanged MambaIRv2 backbone:
    ``F_rdt = (1 + gamma) * F_rd + beta``.
    """

    def __init__(self,
                 clip_model_path=None,
                 clip_freeze=True,
                 clip_local_files_only=True,
                 max_text_length=77,
                 **kwargs):
        if not clip_model_path:
            raise ValueError('RGBDepthTextMambaIRv2 requires clip_model_path.')
        if AutoTokenizer is None or CLIPTextModel is None:
            raise ImportError('Install transformers and tokenizers to use Stage-1 B3.')

        # B3 change: constructing B1 first preserves its complete parameter
        # initialization order; the zero-initialized text branch is added last.
        super().__init__(**kwargs)
        self.clip_model_path = clip_model_path
        self.clip_freeze = bool(clip_freeze)
        self.max_text_length = int(max_text_length)
        self.clip_tokenizer = AutoTokenizer.from_pretrained(
            clip_model_path, local_files_only=bool(clip_local_files_only))
        self.clip_text_encoder = CLIPTextModel.from_pretrained(
            clip_model_path, local_files_only=bool(clip_local_files_only))
        clip_hidden_size = int(self.clip_text_encoder.config.hidden_size)

        if self.clip_freeze:
            for parameter in self.clip_text_encoder.parameters():
                parameter.requires_grad = False
            self.clip_text_encoder.eval()

        # B3 change: the final projection is zero initialized, so B3 is exactly
        # B1 at iteration zero and learns only the incremental value of text.
        self.text_film = nn.Sequential(
            nn.LayerNorm(clip_hidden_size),
            nn.Linear(clip_hidden_size, self.embed_dim * 2))
        self.text_film.apply(self._init_weights)
        nn.init.zeros_(self.text_film[-1].weight)
        nn.init.zeros_(self.text_film[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        if self.clip_freeze:
            self.clip_text_encoder.eval()
        return self

    def state_dict(self, *args, **kwargs):
        """Exclude reproducibly reloadable frozen CLIP weights from checkpoints."""
        state = super().state_dict(*args, **kwargs)
        # B3 change: CLIP is reconstructed from clip_model_path instead of
        # duplicating its frozen weights in every 5k-iteration checkpoint.
        for key in list(state.keys()):
            if key.startswith('clip_text_encoder.'):
                del state[key]
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Keep strict loading for B3 weights while allowing omitted CLIP keys."""
        incompatible = super().load_state_dict(state_dict, strict=False)
        missing_required = [
            key for key in incompatible.missing_keys
            if not key.startswith('clip_text_encoder.')]
        unexpected = list(incompatible.unexpected_keys)
        if strict and (missing_required or unexpected):
            raise RuntimeError(
                'B3 checkpoint mismatch. Missing non-CLIP keys: '
                f'{missing_required}; unexpected keys: {unexpected}.')
        return type(incompatible)(missing_required, unexpected)

    def encode_text(self, text, batch_size, device, dtype):
        if text is None:
            raise ValueError('RGBDepthTextMambaIRv2 requires one caption per image.')
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, (list, tuple)):
            raise TypeError(f'text must be a string, list, or tuple, got {type(text).__name__}.')
        if len(text) != batch_size:
            raise ValueError(f'Text batch size {len(text)} does not match image batch size {batch_size}.')
        descriptions = [str(item).strip() for item in text]
        if any(not item for item in descriptions):
            raise ValueError('Stage-1 B3 does not accept empty captions.')

        clip_limit = int(getattr(self.clip_text_encoder.config, 'max_position_embeddings', 77))
        tokenized = self.clip_tokenizer(
            descriptions,
            padding='max_length',
            truncation=True,
            max_length=min(self.max_text_length, clip_limit),
            return_tensors='pt')
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        if self.clip_freeze:
            self.clip_text_encoder.eval()
            with torch.no_grad():
                encoded = self.clip_text_encoder(**tokenized)
        else:
            encoded = self.clip_text_encoder(**tokenized)
        if encoded.pooler_output is None:
            raise RuntimeError('CLIPTextModel did not return pooler_output.')
        return self.text_film(encoded.pooler_output).to(dtype=dtype)

    def forward(self, rgb, depth, text=None, text_condition=None):
        if depth is None:
            raise ValueError('RGBDepthTextMambaIRv2 requires a depth tensor.')
        if rgb.ndim != 4 or depth.ndim != 4:
            raise ValueError('RGB and depth inputs must both be BCHW tensors.')
        if rgb.shape[0] != depth.shape[0] or rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError(f'RGB/depth batch or spatial mismatch: {rgb.shape} vs {depth.shape}.')
        if depth.shape[1] != self.depth_in_chans:
            raise ValueError(f'Expected {self.depth_in_chans} depth channel(s), got {depth.shape[1]}.')

        h_ori, w_ori = rgb.shape[-2:]
        h = ((h_ori + self.window_size - 1) // self.window_size) * self.window_size
        w = ((w_ori + self.window_size - 1) // self.window_size) * self.window_size
        rgb = self._mirror_pad_to_window(rgb, h, w)
        depth = self._mirror_pad_to_window(depth, h, w)

        self.mean = self.mean.type_as(rgb)
        rgb = (rgb - self.mean) * self.img_range
        depth = depth.type_as(rgb)
        attn_mask = self.calculate_mask([h, w]).to(rgb.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        rgb_feature = self.conv_first(rgb)
        depth_feature = self.depth_conv(depth)
        fused = self.rgb_depth_fusion(torch.cat([rgb_feature, depth_feature], dim=1))
        if text_condition is None:
            text_condition = self.encode_text(
                text, rgb.shape[0], rgb.device, fused.dtype)
        if text_condition.shape != (rgb.shape[0], self.embed_dim * 2):
            raise ValueError(
                f'Expected text_condition shape {(rgb.shape[0], self.embed_dim * 2)}, '
                f'got {tuple(text_condition.shape)}.')
        gamma, beta = text_condition.chunk(2, dim=1)
        modulated = (
            fused * (1.0 + gamma[:, :, None, None])
            + beta[:, :, None, None])
        body = self.conv_after_body(self.forward_features(modulated, params)) + modulated

        if self.upsampler == 'pixelshuffle':
            output = self.conv_last(self.upsample(self.conv_before_upsample(body)))
        elif self.upsampler == 'pixelshuffledirect':
            output = self.upsample(body)
        elif self.upsampler == 'nearest+conv':
            output = self.conv_before_upsample(body)
            output = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(output, scale_factor=2, mode='nearest')))
            output = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(output, scale_factor=2, mode='nearest')))
            output = self.conv_last(self.lrelu(self.conv_hr(output)))
        else:
            output = rgb + self.conv_last(body)

        output = output / self.img_range + self.mean
        return output[..., :h_ori * self.upscale, :w_ori * self.upscale]
