from TrainAndTest import TrainerAndTester
from utils.ClassifierOptions import Options
import os

globalVars = Options()
Vars = globalVars.parse()

trainer = TrainerAndTester(Vars)

trainer.test('./saved_models/resnet18_4_2_20.pt')
