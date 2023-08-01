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


#get_args taken from surface water classifier

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', action = 'store', 
                        dest='model', default='model.pt', 
                        help='path to model')
    parser.add_argument('-stream', dest='stream',
                action='store', default="left_camera",
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
    image =sample.data
    timestamp = sample.timestamp
    transformation = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                         ])
    image = transformation(image)


    image = image.to(args.device).unsqueeze(0)



    with torch.no_grad():
    	output = model(image)




    #inference

    predicted_irradiance = output.item()

    print(f"Current Solar Irradiance: {predicted_irradiance}")


    plugin.publish('env.solar.irradiance', predicted_irradiance, timestamp = timestamp)



if __name__ == "__main__":
    args = get_args()
    if torch.cuda.is_available():
        args.device = 'cuda:0'
    else:
        args.device = 'cpu'

    # model_dict = torch.load(args.model, map_location= args.device)
    # model = torchvision.models.load_state_dict(model_dict)
    # model.eval()

    model = torchvision.models.resnet50()
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 1)

    model_dict = torch.load(args.model, map_location = args.device)
    model.load_state_dict(model_dict)
    model = model.to(args.device)

    model.eval()
    while True:
        with Plugin() as plugin, Camera(args.stream) as camera:
            sample = camera.snapshot()
            run(model,sample, plugin)
            if not args.continuous:
            	exit(0)

