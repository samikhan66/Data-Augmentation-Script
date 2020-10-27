# Data-Augmentation-Script
This script is used to create additional images for your training/test set if you have the bottleneck of less training data.

Step 1: Crop your boundary boxes (that exist in xml files) using crop_images.py. This script will create the objects to be inserted in your synthetic image.


Step 2: Run create.py, this will use the cropped objects and insert them at random places in order to generate a newly synthetic image. The best part about this script is that it will create or update a new xml file that has the xml boundary boxes which means that the new objects will not overlap with the existing coordinates while generating xmin, ymin, xmax, ymax so that no re-annotation has to be done. This can generate synethetic images, hence, no need to wait for more training data and saving countless hours in annotation.
