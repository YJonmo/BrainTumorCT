####################################################################################################
# this is sample call for running the trained network on the dicom scans located in test_data folder
####################################################################################################

# from utils.ClassifierOptions import Options
from torchvision import transforms, models
# from TrainAndTest import TrainerAndTester
from utils.preprocess_funcs import *
import torch.nn as nn
import torch
import glob
import cv2
import os

# globalVars = Options()
# Vars = globalVars.parse()
# tester = TrainerAndTester(Vars)

# Labels (classes)
Brain_dignosis = ['No Tumor', 'Tumor']
no_classes = 2  # number of the classes it is trained for tumor and no tumore

# Path to the dicom data
CTs = sorted(glob.glob('/content/drive/MyDrive/BrainTumorCT2/test_data' + '/*'))

# Creating the model
testingModel = models.resnet50()
n_inputs = testingModel.fc.in_features

testingModel.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, no_classes),
                                nn.LogSigmoid())

# Loading the pretrained weights on the training_data.pickle
testingModel.load_state_dict(torch.load('/content/drive/MyDrive/BrainTumorCT2/saved_models/resnet50_4_2_40.pt'))

# testingModel = tester.get_model(pretrained_path=os.path.join(Vars.save_path, 'resnet50_4_2_40.pt'))
device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

testingModel.to(device)
Trans = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor()])

#Running the loop for classification
with torch.no_grad():
    for CT in CTs:
        frame = remove_noise(CT)
        frame = cv2.resize(frame.astype('uint8'), (128, 128), interpolation=cv2.INTER_AREA)
        frame_0 = torch.zeros((1, 3, frame.shape[0], frame.shape[1]), device=device, requires_grad=False)
        frame_0[0, 0:3, :, :] = torch.tensor(frame).data
        y_val = testingModel(frame_0)
        # The argmax of the predicted tensor is assigned as our label
        pred_label = torch.argmax(y_val, dim=1).data
        print(Brain_dignosis[pred_label] + ' is detected in this scan: ' + CT.split(os.path.sep)[-1])

