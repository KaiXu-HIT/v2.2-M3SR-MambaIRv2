from collections import OrderedDict

import torch
from torch.nn import functional as F

from basicsr.models.mambairv2_model import MambaIRv2Model
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class RGBDepthTextMambaIRv2Model(MambaIRv2Model):
    """Stage-1 B3 training and tiled-evaluation wrapper."""

    def setup_optimizers(self):
        # B3 change: frozen CLIP tensors are excluded from Adam and summarized
        # once, while every RGB-depth backbone and FiLM tensor stays trainable.
        train_opt = self.opt['train']
        trainable = [parameter for parameter in self.net_g.parameters() if parameter.requires_grad]
        trainable_count = sum(parameter.numel() for parameter in trainable)
        frozen_count = sum(
            parameter.numel() for parameter in self.net_g.parameters()
            if not parameter.requires_grad)
        get_root_logger().info(
            f'B3 optimizer parameters: {trainable_count:,d} trainable; '
            f'{frozen_count:,d} frozen CLIP parameters.')
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, trainable, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.depth = data['depth'].to(self.device)
        self.text = data['text']
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.depth, text=self.text)

        l_total = 0
        loss_dict = OrderedDict()
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        _, channels, h, w = self.lq.size()
        split_token_h = h // 200 + 1
        split_token_w = w // 200 + 1
        mod_pad_h = (split_token_h - h % split_token_h) % split_token_h
        mod_pad_w = (split_token_w - w % split_token_w) % split_token_w
        rgb = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        depth = F.pad(self.depth, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, padded_h, padded_w = rgb.size()
        split_h = padded_h // split_token_h
        split_w = padded_w // split_token_w
        shave_h = split_h // 10
        shave_w = split_w // 10
        scale = self.opt.get('scale', 1)

        slices = []
        for i in range(split_token_h):
            for j in range(split_token_w):
                top_start = i * split_h if i == 0 else i * split_h - shave_h
                top_end = (i + 1) * split_h if i == split_token_h - 1 else (i + 1) * split_h + shave_h
                left_start = j * split_w if j == 0 else j * split_w - shave_w
                left_end = (j + 1) * split_w if j == split_token_w - 1 else (j + 1) * split_w + shave_w
                slices.append((slice(top_start, top_end), slice(left_start, left_end)))

        test_net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.get_bare_model(self.net_g)
        was_training = test_net.training
        test_net.eval()
        with torch.no_grad():
            # B3 change: encode one global caption once and reuse its FiLM
            # condition for every aligned RGB/depth tile.
            text_condition = test_net.encode_text(
                self.text, rgb.shape[0], rgb.device, rgb.dtype)
            outputs = [
                test_net(
                    rgb[..., top, left], depth[..., top, left],
                    text_condition=text_condition)
                for top, left in slices]
            merged = torch.zeros(
                (rgb.shape[0], channels, padded_h * scale, padded_w * scale),
                device=outputs[0].device, dtype=outputs[0].dtype)
            for i in range(split_token_h):
                for j in range(split_token_w):
                    top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                    left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                    source_top = slice(0, split_h * scale) if i == 0 else slice(shave_h * scale, (shave_h + split_h) * scale)
                    source_left = slice(0, split_w * scale) if j == 0 else slice(shave_w * scale, (shave_w + split_w) * scale)
                    merged[..., top, left] = outputs[i * split_token_w + j][..., source_top, source_left]
        if was_training:
            test_net.train()
        self.output = merged[..., :h * scale, :w * scale]

    def get_current_visuals(self):
        visuals = super().get_current_visuals()
        visuals['depth'] = self.depth.detach().cpu()
        return visuals
