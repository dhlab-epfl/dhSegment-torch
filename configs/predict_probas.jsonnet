local train = import 'train.jsonnet';

# Config to predict probabilties maps (numpy format) using the predict_probas.py script.
# Most of the configuration is imported from the train.jsonnet

{
  # TODO complete the path to the folder containing the images to be predicted
  data: {
    type: "folder",
    folder: "/path/to/images/to/predict/folder",
    # Copy the same pre-processing as validation
    pre_processing: train.val_dataset.compose
  },

  # Batch size copied from training
  batch_size: train.batch_size,

  # Model imported from training
  model: {
    type: "training_config",
    model: train.model,
    color_labels: train.color_labels,
    dataset: train.train_dataset
  }
}