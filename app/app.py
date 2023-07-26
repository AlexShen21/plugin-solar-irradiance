from waggle.plugin import Plugin
from waggle.data.vision import Camera
import time

import torch
from torchvision import transforms

import time
import cv2
from PIL import Image
import argparse



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', action = 'store', 
                        dest='model', default='model.pth', 
                        help='path to model')
    parser.add_argument('-stream', dest='stream',
                action='store', default="bottom",
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



    with torch.no_grad():
    	output = model(image)


    #inference

    predicted_irradiance = output.item()

    if args.debug:
    	print(f"Current Solar Irradiance: {predicted_irradiance}")


    plugin.publish('env.solar.irradiance', predicted_irradiance, timestamp = timestamp)



if __name__ == "__main__":
    args = get_args()
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    model = torch.load(args.model, map_location=args.device)
    model.eval()
    while True:
        with Plugin() as plugin, Camera(args.stream) as camera:
            sample = camera.snapshot()
            run(model,sample, plugin)
            if not args.continuous:
            	break
            	exit(0)

