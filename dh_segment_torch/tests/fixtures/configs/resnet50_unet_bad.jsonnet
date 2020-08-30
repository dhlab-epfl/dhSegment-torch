local resnet = import 'resnet.libsonnet';
local unet = import 'unet.libsonnet';

{
  "encoder": resnet.resnet50,
  "decoder": unet
  }