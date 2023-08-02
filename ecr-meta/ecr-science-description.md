# Science
Estimates solar irradiance based on ground images of the sky taken from SAGE waggle nodes. The app was created from a pretrained Resnet50 model. Hopefully this solar irradiance plugin will be able to with solar energy generation, climate change, weather forecasting, and integrated into smart homes to control certain appliances.

# AI at the Edge
The Resnet 50 model was fine tuned and trained on a dataset of over 4000 images that was collected from the Sage Waggle node. The images were not augmented in any way so there is room for improvement. While the plugin runs, the model takes in an image that is a snapshot taken from a given top camera which is installed at the sage node. The model then performs inference and gives an estimate of the approximate solar irradiance from the image.

# Ontology
The code publishes measurement with topic env.solar.irradiance



