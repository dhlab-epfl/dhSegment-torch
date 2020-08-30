local resnet = import 'resnet.libsonnet';
local unet = import 'unet.libsonnet';

{
  "encoder": resnet.resnet50,
  "decoder": unet {
        decoder_channels: [512, 256, 128, 64, 32]
    },
    "loss" : "cross_entropy",
  }