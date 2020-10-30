# Configuration file to use with the prepare_data.py script.
# Reads a VIA2 file and creates image masks.

{
  # TODO complete the path to the via file, the path to the images (if not using IIIF) and the attribute name
  annotation_reader: {
    type: 'via2_project', # File format of the data
    attrib_name: 'lines', # Name of the via attribute where the labels are defined
    file_path: '/path/to/the/annotation/file.json', # Path to the annotation file
    images_dir: '/path/to/the/images', # Path to the images directory (not necessary if using IIIF)
    line_thickness: 4, # Thickness of the lines to draw
  },

  # TODO complete the mapping from label to color
  color_labels: {
    type: labels_list,
    color_labels: [
        {color: "#FF0000",
         label: "class1"},
    ]
  },

  # Path to write the data
  data_path : '/path/to/output/data',
  # Splitting ratio of the data, here we have not test set since we have little data
  data_splitter: {'train_ratio': 0.8, 'val_ratio': 0.2, 'test_ratio': 0.0},


  # Some options that can be kept as default
  copy_images: True, # Whether to copy the images
  overwrite: True, # Whether to overwrite the images
  progress: True, # Whether to show progress
  resizer: {height: 2000} # Size to which the images should be resized while importing them (useful, since we resize them anyway in the processing pipeline)
}