####################################################################################################
# this is a sample call to train the network on the already created pickled files of the brain scans
####################################################################################################
from TrainAndTest import TrainerAndTester
from utils.ClassifierOptions import Options

globalVars = Options()
Vars = globalVars.parse()

trainer = TrainerAndTester(Vars)
trainer.train()
