import os
import gdown
from pathlib import Path
from utils.ClassifierOptions import Options

globalVars = Options()
Vars = globalVars.parse()


if not os.path.exists(Vars.data_path):
    os.mkdir(Vars.data_CTtraining)

gdown.download('https://drive.google.com/uc?export=download&id=1DXBQfRB2aczpSX_GUyRr4TxE7bflpc8g', os.path.join(globalVars.data_path, 'resnet50_4_2_40.pt'))
gdown.download('https://drive.google.com/uc?export=download&id=1VTUXanL66UL_YNN9twaWX1a0nspEhQb2', os.path.join(globalVars.data_path, 'training_data.pkl'))
gdown.download('https://drive.google.com/uc?export=download&id=18oK4RWmfACHVo6PtbZshYcBxiIiDC65L', os.path.join(globalVars.data_path, 'testing_data.pkl'))

#ROOT = os.getcwd()

#Path(os.path.join(globalVars.data_path, 'resnet50_4_2_40.pt')).rename(  ,resnet50_4_2_40.pt))