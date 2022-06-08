import os
import argparse

#file_dir = os.path.dirname(__file__)  # the directory that this file resides in
ROOT = os.getcwd() # root directory of the repo

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="CT Tumor Classifier Options")

        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to all of the data",
                                 default=os.path.join(ROOT, 'data_CTtraining'))

        self.parser.add_argument("--data_dicom",
                                 type=str,
                                 help="path to the raw dicom data",
                                 default=os.path.join(ROOT, 'data_CTtraining'))

        self.parser.add_argument("--data_jpg",
                                 type=str,
                                 help="path to the preprocessed data",
                                 default=os.path.join(ROOT, 'data_CTtraining', 'data_jpg'))


        self.parser.add_argument("--CSV_path_orig",
                                 type=str,
                                 help="path to the raw data labels",
                                 default=os.path.join(ROOT, 'data_CTtraining', 'annotations.csv'))

        self.parser.add_argument("--CSV_path_aug",
                                 type=str,
                                 help="path to the raw data labels",
                                 default=os.path.join(ROOT, 'data_CTtraining', 'annotations_aug.csv'))

        self.parser.add_argument("--pickle_path_train",
                                 type=str,
                                 help="path to the training pickled data",
                                 default=os.path.join(ROOT, 'data_CTtraining', 'training_data.pickle'))

        self.parser.add_argument("--pickle_path_test",
                                 type=str,
                                 help="path to the testing pickled data",
                                 default=os.path.join(ROOT, 'data_CTtraining', 'testing_data.pickle'))

        self.parser.add_argument("--save_path",
                                 type=str,
                                 help="path to the saved models",
                                 default=os.path.join(ROOT, 'saved_models'))

        self.parser.add_argument("--saved_model_name",
                                 type=str,
                                 help="name of the saved model for validation",
                                 default=os.path.join(ROOT, 'saved_models/resnet50_4_2_40.pt'))

        self.parser.add_argument("--test_dicom",
                                 type=str,
                                 help="path to the saved models",
                                 default=os.path.join(ROOT, 'test_data'))


        self.parser.add_argument("--no_classes",
                                 type=int,
                                 help="number of the classes for classification. Options are 2 to 6. The is healthy versus tumor is 2",
                                 default=2)

        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="size of the batch",
                                 default=4)

        self.parser.add_argument("--epochs",
                                 type=int,
                                 help="number of epochs for training",
                                 default=30)

        self.parser.add_argument("--train2test_ratio",
                                 type=float,
                                 help="training to testing split ratio. It only works when the pickel file being created, otherwise it will be 0.7",
                                 default=0.7)

        self.parser.add_argument("--image_size",
                                 type=int,
                                 help="image size in pixels given to the network",
                                 default=128)

        self.parser.add_argument("--model",
                                 type=str,
                                 help="the model chosen from pytorch models. Options are resnet18, resnet50, and resnet152,",
                                 default='resnet50')

        self.parser.add_argument("--imageSizeThreshMaxMin",
                                 type=int,
                                 help="the maximum and minimum sizes for rejecting a frame as a non brain frame,",
                                 default=[131430, 21111])


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
