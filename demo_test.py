####################################################################################################
# this is sample call for running the trained network on the already created pickle files of the brain scans
####################################################################################################

from TrainAndTest import TrainerAndTester
from utils.ClassifierOptions import Options
import os

globalVars = Options()
Vars = globalVars.parse()

trainer = TrainerAndTester(Vars)

trainer.test(os.path.join(Vars.save_path  , 'resnet50_4_2_40.pt'))
