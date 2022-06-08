####################################################################################################
# this is pipeline for reading, preprocessing, and converting the dicom images into jpg and pickle
# files and splitting them into training and testing
# This is a time consuming and the processed pickled outputs are provided via google drive link
####################################################################################################
from utils.preprocess_funcs import *
from utils.ClassifierOptions import Options
from utils.CT_preprocess import CT_preprocess

globalVars = Options()
Vars = globalVars.parse()

# creating a class for processing the data
preoces_data = CT_preprocess(Vars)
#if the data already existing in the pickle format then you do not need to run this code
# step 1: download the zip file
preoces_data.download_dicom()

# step 2: extract the dicom file
preoces_data.extract_dicom()

# step 3: preprocess the dicom, augment the tumor data (to balance the ratio), 3 save the result in JPG format
preoces_data.do_process()

# step 4: read the JPG processed data and create training and testing pickle files
preoces_data.create_pickle()
