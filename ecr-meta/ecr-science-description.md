# Science
Estimates solar irradiance based on ground images of the sky taken from SAGE waggle nodes. The app was created from a pretrained Resnet50 model with a RMSE loss function to predict a continuous value. The model was trained on over 4000 images which was taken from a csv file that was also created which paired irradiance values with their respective images. The model was finetuned and trained using residual learning with preset weights and the last layer was finetuned on training dataset.


