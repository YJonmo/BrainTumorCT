####################################################################################################
# this is sample call for running the trained network on the already created pickle files of the brain scans
####################################################################################################

from utils.ClassifierOptions import Options
from TrainAndTest import TrainerAndTester
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import os

globalVars = Options()
Vars = globalVars.parse()

tester = TrainerAndTester(Vars)

#runing the trained model on testing data
tester.test(os.path.join(Vars.save_path  , 'resnet50_4_2_40.pt'))
