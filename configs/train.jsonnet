local train_base = import 'train_base.libsonnet';


# Read the comments and change at least the values with a TODO

# Used to change the size of the resize of our images.
# The goal is to find the smallest resolution where your classes are stil discernable.
# You should not go below 1e5 in most cases, however, you can try 5e5 depending on your class.
local fixed_size_resize = 1e6;

train_base + {

  # TODO update the path to which the model will be saved
  model_out_dir : "/path/to/model/dir",

  # Definition of the mapping of colors and labels
  # Need to at least define each color (label key is optional)
  # TODO update the colors_labels to match your own data.
  color_labels: {
    type: "labels_list",
    color_labels: [
        # Note that we need to add the background as a class.
        {color: "#000000",
         label: "background"},
        {color: "#FF0000",
         label: "class1"},
    ]
  },
  # If you used the data creation script, you can use the generated json file.
  # You can comment the previous color_labels and uncomment following lines
  # color_labels: {
  #   type: "json",
  #   label_json_file: "/path/to/color_labels.json"
  #},

  # Definition of the training and validation datasets
  # TODO update the paths to the CSVs and base_dir

  # Need to have CSVs with two columns and NO HEADER, containing paths to the image and the image.
  # Paths can be absolute and the base_dir should be removed.
  # If paths are relative, they should be relative towards base_dir

  train_dataset: {
         type: "image_csv",
         csv_filename: "/path/to/train.csv",
         base_dir: "/path/to/images/and/labels/directory",
         # Number of time we repeat the dataset, useful when you have a small dataset to train a bit faster.
         repeat_dataset: 3,
         compose: {transforms:
            [{type: "fixed_size_resize", output_size: fixed_size_resize}] # Resize to a fixed size
            # Several other transformations are added to be used for data augmentation
            # They may not all make sense with your data, use them carefully
            }

     },
  # Same as training without data augmentation, but keeping the fixed resize
  val_dataset: {
         type: "image_csv",
         csv_filename: "/path/to/val.csv",
         base_dir: "/path/to/images/and/labels/directory",
         compose: {transforms: [{type: "fixed_size_resize", output_size: fixed_size_resize}]}
     },

  # Following are the main training parameters that should be updated.
  # You can try with the defaults and then change them according to the results.

  # The number of epochs (full iteration on the training dataset) the network will train.
  # It can be set to a lower value if the network converges quickly.
  num_epochs: 100,

  # The learning rate of the optimizer.
  # If the learning is "noisy", i.e. the loss varies too much, it should be decreased
  # If the learning is too "flat", i.e. the loss stagnates too much, it should be increased
  optimizer: {
        lr: 1e-3
  },
  # The number of samples to show at each iteration.
  # Usually, we want this number to be as big as the GPU memory allows us to have it.
  # Try to increase it until you get an error stating that you are using too much GPU, then take the last working value
  # The GPU memory usage depends on this parameter and on the fixed_size_resize parameter.
  batch_size: 4,

  # The number of epochs to train before running an evalution.
  # It allows to keep track of the progress of the network and saves the network when a new best score is reached.
  # However, it slows down the training.
  # It should be set according to your dataset size and the noisiness of the learning processs.
  evaluate_every_epoch: 2,

  # Once the network was evaluated n times without improvement, the training will automatically stop.
  early_stopping: {
    patience: 5
  },
}
