{
    // Color labels file
    "color_labels": {
        "label_json_file": "data/my_line_seg_proj/color_labels.json",
    },

    // Training dataset
    "train_dataset": {
        "type": "image_csv",
        "csv_filename": "data/my_line_seg_proj/train.csv",
        "base_dir": ".", // Paths in CSV are already absolute
        "repeat_dataset": 1,
        "compose": {
            "transforms": [
                { "type": "random_shadow", "p": 0.2 },
                { "type": "vertical_flip", "p": 0.3 },
                { "type": "blur", "p": 0.2, "blur_limit": 3 }
            ]
        }
    },

    // Validation dataset
    "val_dataset": {
        "type": "image_csv",
        "csv_filename": "data/my_line_seg_proj/val.csv",
        "base_dir": ".", // Paths in CSV are already absolute
        "compose": { "transforms": [] }
    },

    // Model definition
    "model": {
        "encoder": "resnet101",
        "decoder": {
            "decoder_channels": [512, 256, 128, 64, 32],
            "max_channels": 512
        }
    },

    // Training parameters
    "optimizer": { "type": "Adam", "lr": 1e-3 },
    "lr_scheduler": { "type": "exponential", "gamma": 0.9995 },
    "val_metric": "+miou",
    "early_stopping": { "patience": 25 },
    "model_out_dir": "models/my_line_seg_proj/", // Where to save your trained model
    "num_epochs": 200, // You can start with 200 and see how it goes
    "evaluate_every_epoch": 1,
    "batch_size": 1, // Keep this at 1 unless you have a very powerful GPU
    "num_data_workers": 2,
    "track_train_metrics": false,

    // Logging
    "loggers": [
        {
            "type": "tensorboard",
            "log_dir": "runs/my_line_seg_proj/log/",
            "log_every": 4, "log_images_every": 60
        }
    ]
}