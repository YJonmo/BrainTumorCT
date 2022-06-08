####################################################################################################
# this is sample call for running the trained network on the already created pickle files of the brain scans
####################################################################################################

from utils.ClassifierOptions import Options
from TrainAndTest import TrainerAndTester
from utils.preprocess_funcs import *
import torch
import glob
import cv2
import os

globalVars = Options()
Vars = globalVars.parse()
tester = TrainerAndTester(Vars)

Brain_dignosis = ['No Tumor', 'Tumor']

CTs = sorted(glob.glob(Vars.test_dicom + '/*'))



testingModel = tester.get_model()
testingModel.to(tester.device)
Trans = T.Compose([T.ToPILImage(), T.Resize((128, 128)),T.ToTensor()])
with torch.no_grad():
    for CT in CTs:
        frame = remove_noise(CT)
        frame = cv2.resize(frame.astype('uint8'), (Vars.image_size, Vars.image_size), interpolation=cv2.INTER_AREA)
        frame_0 = torch.zeros((1,3, frame.shape[0], frame.shape[1]), device=tester.device, requires_grad=False)
        frame_0[0,0:3,:,:] = torch.tensor(frame).data
        y_val = testingModel(frame_0)
        # The argmax of the predicted tensor is assigned as our label
        pred_label = torch.argmax(y_val, dim=1).data
        print(Brain_dignosis[pred_label] + ' is detected in this scan: ' + CT.split(os.path.sep )[-1])



trainer = TrainerAndTester(Vars)

trainer.test(os.path.join(Vars.save_path  , 'resnet50_4_2_40.pt'))
