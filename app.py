from waggle.plugin import Plugin
from waggle.data.vision import Camera
import time
import torchvision.models
import torch
from torchvision import transforms
import torch.nn

import time
import cv2
from PIL import Image
import argparse


#get_args function taken from surface water classifier
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', action = 'store', 
                        dest='model', default='model.pt', 
                        help='path to model')
    parser.add_argument('-stream', dest='stream',
                action='store', default="top_camera",
                help='ID or name of a stream, e.g. sample')
    parser.add_argument(
        '-debug', dest='debug',
        action='store_true', default=False,
        help='Debug flag')
    parser.add_argument(
        '-continuous', dest='continuous',
        action='store_true', default=False,
        help='Continuous run flag')
    return parser.parse_args()


def run(model, sample, plugin):
    #Take the image sample from camera
    image =sample.data

    #Get the timestamp 
    timestamp = sample.timestamp

    #Transform to PIL Image then resize to the size of images that model was trained on
    transformation = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                         ])
    image = transformation(image)


    
    #Use either Cuda or CPU
    image = image.to(args.device).unsqueeze(0)


    #Perform Inference
    with torch.no_grad():
    	output = model(image)



    #Store inference
    predicted_irradiance = output.item()

    #Print the predicted irradiance to logs
    print(f"Current Solar Irradiance: {predicted_irradiance}")

    #Publish the estimated irradiance data
    plugin.publish('env.solar.irradiance', predicted_irradiance, timestamp = timestamp)



if __name__ == "__main__":
    
    args = get_args()

    #Set the device
    if torch.cuda.is_available():
        args.device = 'cuda:0'
    else:
        args.device = 'cpu'
    
    #Load saved model from LCRC
    model = torchvision.models.resnet50()
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 1)

    model_dict = torch.load(args.model, map_location = args.device)
    model.load_state_dict(model_dict)
    model = model.to(args.device)

    model.eval()

    #Use Camera to get image and run it through the above functions
    while True:
        with Plugin() as plugin, Camera(args.stream) as camera:
            sample = camera.snapshot()
            run(model,sample, plugin)
            if not args.continuous:
            	exit(0)

