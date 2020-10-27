# Data-Augmentation-Script
This script is used to create additional images for your training/test set if you have the bottleneck of less training data.

Step 1: Crop your boundary boxes (that exist in xml files) using crop_images.py. This script will create the objects to be inserted in your synthetic image.
Step 2: Run create.py, this will use the cropped objects and insert them at random places in order to generate a newly synthetic image.
