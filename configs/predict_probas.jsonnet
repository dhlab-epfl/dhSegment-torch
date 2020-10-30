local train = import 'train.jsonnet';

# Config to predict probabilties maps (numpy format) using the predict_probas.py script.
# Most of the configuration is imported from the train.jsonnet

{
  output_directory: "/path/where/to/output/probas",

  # TODO complete the path to the folder containing the images to be predicted
  data: {
    type: "folder",
    folder: "/path/to/images/to/predict/folder",
    # Copy the same pre-processing as validation
    pre_processing: train.val_dataset.compose
  },


  # Model imported from training
  # TODO add the path to the best model
  model: {
    type: "training_config",
    model: train.model,
    color_labels: train.color_labels,
    dataset: train.train_dataset,
    model_state_dict: "/path/to/best_model_miou=.pth"
  },

  # Batch size copied from training
  batch_size: train.batch_size,

}