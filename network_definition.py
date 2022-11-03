"""
Given the L channel of an Lab image (range [-1, +1]), output a prediction over
the a and b channels in the range [-1, 1].
In the neck of the conv-deconv network use the features from a feature extractor
(e.g. Inception) and fuse them with the conv output.
"""

from keras.engine import InputLayer
from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential

import torch
import torch.nn as nn

from .fusion_layer import FusionLayer


class Colorization:
    def __init__(self, depth_after_fusion):
        self.encoder = _build_encoder()
        self.fusion = FusionLayer()
        self.after_fusion = nn.Conv2D(depth_after_fusion, (1, 1))
        self.after_fusion = nn.ReLU(inplace=True)
        self.decoder = _build_decoder(depth_after_fusion)

    def build(self, img_l, img_emb):
        img_enc = self.encoder(img_l)

        fusion = self.fusion([img_enc, img_emb])
        fusion = self.after_fusion(fusion)

        return self.decoder(fusion)


def _build_encoder():
    model = nn.Sequential(name="encoder")
    model.append(nn.InputLayer(input_shape=(None, None, 1)))
    model.append(nn.Conv2D(64, (3, 3), padding="same", strides=2))
    model.append(nn.ReLU(inplace=True))
    model.append(nn.Conv2D(128, (3, 3), padding="same"))
    model.append(nn.ReLU(inplace=True))
    model.append(nn.Conv2D(128, (3, 3), padding="same", strides=2))
    model.append(nn.ReLU(inplace=True))
    model.append(nn.Conv2D(256, (3, 3), padding="same"))
    model.append(nn.ReLU(inplace=True))
    model.append(nn.Conv2D(256, (3, 3), padding="same", strides=2))
    model.append(nn.ReLU(inplace=True))
    model.append(nn.Conv2D(512, (3, 3), padding="same"))
    model.append(nn.ReLU(inplace=True))
    model.append(nn.Conv2D(512, (3, 3), padding="same"))
    model.append(nn.ReLU(inplace=True))
    model.append(nn.Conv2D(256, (3, 3), padding="same"))
    model.append(nn.ReLU(inplace=True))
    return model


def _build_decoder(encoding_depth):
    model = nn.Sequential(name="decoder")
    model.append(nn.UpsamplingNearest2d(input_shape=(None, None, encoding_depth)))
    model.append(nn.Conv2D(128, (3, 3), padding="same"))
    model.append(nn.ReLU(inplace=True))
    model.append(nn.UpsamplingNearest2d((2, 2)))
    model.append(nn.Conv2D(64, (3, 3), padding="same"))
    model.append(nn.ReLU(inplace=True))
    model.append(nn.Conv2D(64, (3, 3), padding="same"))
    model.append(nn.ReLU(inplace=True))
    model.append(nn.UpsamplingNearest2d((2, 2)))
    model.append(nn.Conv2D(32, (3, 3), padding="same"))
    model.append(nn.ReLU(inplace=True))
    model.append(nn.Conv2D(2, (3, 3), padding="same"))
    model.append(nn.Tanh(inplace=True))
    model.append(nn.UpsamplingNearest2d((2, 2)))
    return model