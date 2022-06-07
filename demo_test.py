from TrainAndTest import TrainerAndTester
from utils.ClassifierOptions import Options
import os

globalVars = Options()
Vars = globalVars.parse()

trainer = TrainerAndTester(Vars)

trainer.test(os.path.join(Vars.save_path  , 'resnet50_4_2_40.pt'))
