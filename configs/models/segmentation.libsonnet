local encoders = import 'encoders.libsonnet';
local decoders = import 'decoders.libsonnet';

{
    resnet_50_unet: {
        encoder: encoders.resnet50,
        decoder: decoders.unet {
            decoder_channels: [512, 256, 128, 64, 32],
            max_channels: 512
            }
        },
    resnet_50_pan: {
        encoder: encoders.resnet50,
        decoder: decoders.pan {
            decoder_channels_size: 512
        }
    },
    mobilenetv2_unet: {
        encoder: encoders.mobilenetv2,
        decoder: decoders.unet {
            decoder_channels: [96, 64, 32, 24, 16],
            max_channels: 512
        }
    }
 }