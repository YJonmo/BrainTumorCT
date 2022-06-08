import os
import cv2
import glob
import gdown
import pickle
from zipfile import ZipFile
from .preprocess_funcs import *


class CT_preprocess():
    def __init__(self, Vars): #raw_data_path, CSV_path_aug):
        self.vars = Vars
        # self.raw_data_path = raw_data_path
        # self.CSV_path_aug = CSV_path_aug
    def download_dicom(self, URL='https://drive.google.com/u/0/uc?id=1qlmzqAQr2miOIAIWjdZSy_g3dRP28b0q&export=download',
                       FileName='CTtraining.zip'):
        #Download it
        CTfile = os.path.join(self.vars.data_path, FileName)
        gdown.download(URL, CTfile)

        # Unzip it
        with ZipFile(CTfile, 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(self.vars.data_path)
            print(self.vars.data_path)
            print('GGGGGGGGGGGGGGGGGGGGGGGGGGGGG')

    def extract_dicom(self, rewrite=False):#, image_size, imageSizeThreshMax, imageSizeThreshMin, rewrite=False):
        if not os.path.exists(self.vars.data_jpg):
            os.mkdir(self.vars.data_jpg)
            self.do_process()
        else:
            if rewrite:
                self.do_process()

    def do_process(self):
        CTs = sorted(glob.glob(self.vars.data_dicom + '/*'))
        Labels = read_lables(self.vars.CSV_path_orig)

        counter = 0
        #reject_list = []
        train_data = []
        aug_labels = ['20', '_20', 'Hf', 'Hf20', 'Hf_20']
        write_lables(self.vars.CSV_path_aug, 'w')
        for frame_ind in range(len(Labels['frame'])):
            frame_path = os.path.join(self.vars.data_dicom, Labels['scan'][frame_ind], Labels['frame'][frame_ind] + '.dcm')
            frame = remove_noise(frame_path)
            if not((self.vars.imageSizeThreshMaxMin[1] >= np.sum(frame != 0)) or (np.sum(frame != 0) >= self.vars.imageSizeThreshMaxMin[0])):
                #reject_list.append(frame_ind)
                jpg_name = Labels['scan'][frame_ind] + '_' + Labels['frame'][frame_ind] + '.jpg'
                save_path = os.path.join(self.vars.data_jpg, jpg_name)
                frame = cv2.resize(frame, (self.vars.image_size*1, self.vars.image_size*1), interpolation=cv2.INTER_AREA)
                cv2.imwrite(save_path, frame)

                write_lables(self.vars.CSV_path_aug, 'a', [Labels['frame'][frame_ind], Labels['scan'][frame_ind], Labels['Any'][frame_ind],
                             Labels['IPH'][frame_ind], Labels['IVH'][frame_ind], Labels['SAH'][frame_ind],
                             Labels['SDH'][frame_ind]])

                if Labels['Any'][frame_ind]:
                    # [frame20, frame_20, frameHf, frameHf20, frameHf_20] = augment_data(frame)
                    augmentedList = augment_data(frame)
                    for index, frame in enumerate(augmentedList):
                        frame_tag = Labels['frame'][frame_ind]+ aug_labels[index]
                        jpg_name = Labels['scan'][frame_ind] + '_' + frame_tag + '.jpg'
                        #train_data.append([aug_frame, 1])
                        save_path = os.path.join(self.vars.data_jpg, jpg_name)
                        cv2.imwrite(save_path, frame)
                        write_lables(self.vars.CSV_path_aug, 'a', [frame_tag, Labels['scan'][frame_ind], Labels['Any'][frame_ind],
                                  Labels['IPH'][frame_ind], Labels['IVH'][frame_ind], Labels['SAH'][frame_ind],
                                  Labels['SDH'][frame_ind]])
            # else:
                #reject_list.append(frame_ind)

    def create_pickle(self):
        Labels = read_lables(self.vars.CSV_path_aug)
        train_test_data = []
        for frame_ind in range(len(Labels['frame'])):
            frame_path = os.path.join(self.vars.data_jpg, Labels['scan'][frame_ind] + '_' +
                                      Labels['frame'][frame_ind] + '.jpg')
            frame = cv2.imread(frame_path)
            # frame = cv2.resize(frame, (globVars.image_size, globVars.image_size), interpolation=cv2.INTER_AREA)
            # img2 = cv2.resize(img, (512,512))
            label = [int(not (Labels['Any'][frame_ind])),
                     Labels['Any'][frame_ind],
                     Labels['IPH'][frame_ind],
                     Labels['IVH'][frame_ind],
                     Labels['SAH'][frame_ind],
                     Labels['SDH'][frame_ind]]
            ID = Labels['frame'][frame_ind]
            train_test_data.append([ID, label, frame])

        data_length = len(train_test_data)
        training_length = [0, round(data_length * self.vars.train2test_ratio)]
        testing_length = [round(data_length * self.vars.train2test_ratio), data_length]

        pickle_out_train = open(self.vars.pickle_path_train, "wb")
        pickle_out_test = open(self.vars.pickle_path_test, "wb")
        pickle.dump(train_test_data[training_length[0]:training_length[1]], pickle_out_train)
        pickle.dump(train_test_data[testing_length[0]:testing_length[1]], pickle_out_test)
        pickle_out_train.close()
        pickle_out_test.close()

    def load_pickel(self):#, pickle_path_train, pickle_path_test):
        train_data = pickle.load(open(self.vars.pickle_path_train, 'rb'))
        test_data = pickle.load(open(self.vars.pickle_path_test, 'rb'))
        return train_data, test_data
