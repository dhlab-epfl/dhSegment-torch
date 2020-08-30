local resnet50_unet = import 'resnet50_unet.jsonnet';
local num_epochs = 100;

{
    model: resnet50_unet {
        decoder+: {
            activation : {type: "leaky_relu", inplace: true},
            normalization: "batch_renorm_2d_drop",
        },
        loss: {
            type: "combined_loss",
            losses: ["dice", "cross_entropy"],
            weights: [1, 2]
        }
    },
    dataset: {
        type: "patches_csv",
        csv_filename: './path/to/data.csv',
        patches_size: 32,
        pre_compose_transform: {
            transforms: [
                "gauss_noise",
                "horizontal_flip",
                "jpeg_compression",
                "random_brightness_contrast"
            ]
        },
        post_compose_transform: {
            transforms: [
                {
                    type: "fixed_size_rotate_crop",
                    limit: 5
                },
                {
                    type: "assign_label_classification",
                    colors_array: [[0, 0, 0], [255, 0, 0], [0, 0, 255]],
                },
            ]
        }
    },
    training: {
        num_epochs: num_epochs
    }
}