local segmentation = import './models/segmentation.libsonnet';


{
    # Model definition
    model: segmentation.resnet_50_unet,

    # Define metrics, miou, iou per class and precision
    metrics: [
        ["miou", "iou"],
        ["iou", {
             "type": "iou",
             "average": null
         }],
         "precision"
        ],

     # Validate on miou
     val_metric: "+miou",

     # Defining the default exponential scheduler
     lr_scheduler: {
         type: "exponential",
         gamma: 0.9995
     },

     # Reduce the default number of checkpoints to keep
     train_checkpoint: {type: "iteration", checkpoints_to_keep: 2},
     val_checkpoint: {checkpoints_to_keep: 2},

     # Default values are good for our purpose
     num_data_workers: 4,
     track_train_metrics: false,
     loggers: [{
         type: "tensorboard",
         log_every: 20,
         log_images_every: 50
     }],
}