####################################################################################################
# this is the steps to download the already processed brain CT images as well as the trained weight
####################################################################################################
import os
import gdown
#from pathlib import Path
from utils.ClassifierOptions import Options

globalVars = Options()
Vars = globalVars.parse()

if not os.path.exists(Vars.data_path):
    os.mkdir(Vars.data_path)

if not os.path.exists(Vars.save_path):
    os.mkdir(Vars.save_path)

gdown.download('https://drive.google.com/uc?export=download&id=18oK4RWmfACHVo6PtbZshYcBxiIiDC65L',
               os.path.join(Vars.save_path, 'resnet50_4_2_40.pt'))
gdown.download('https://drive.google.com/uc?export=download&id=1VTUXanL66UL_YNN9twaWX1a0nspEhQb2',
               os.path.join(Vars.data_path, 'testing_data.pickle'))
gdown.download('https://drive.google.com/uc?export=download&id=1DXBQfRB2aczpSX_GUyRr4TxE7bflpc8g',
               os.path.join(Vars.data_path, 'training_data.pickle'))


gdown.download('https://drive.google.com/u/1/uc?id=1A15YMCS93D6JlNDrw84rbxN9kXnQT3M1&export=download',
               os.path.join(Vars.data_path, 'annotations.csv'))

gdown.download('https://drive.google.com/u/1/uc?id=1E2XrQX9DSXvsJwJDYBZ3AfbCQPKSM0Ye&export=download',
               os.path.join(Vars.data_path, 'annotations_aug.csv'))

# ROOT = os.getcwd()

# Path(os.path.join(globalVars.data_path, 'resnet50_4_2_40.pt')).rename(  ,resnet50_4_2_40.pt))
