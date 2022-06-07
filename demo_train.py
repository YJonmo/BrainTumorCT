from TrainAndTest import TrainerAndTester
from utils.ClassifierOptions import Options

globalVars = Options()
Vars = globalVars.parse()

tester = TrainerAndTester(Vars)
tester.train()
