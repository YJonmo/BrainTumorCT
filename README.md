# This is the demo for performing tumor classification on the brain CT images 

The data is not publically available, as a result only the training pipeline and the a trained model of resnet is provided in this repo.


Intall the required packages using

pip install -r  requirements.txt 

Link for a pretrained network to perform binary classification (tumore versus no tumor):

https://drive.google.com/file/d/18oK4RWmfACHVo6PtbZshYcBxiIiDC65L/view?usp=sharing

Link for the training and testing data:

https://drive.google.com/uc?export=download&id=1DXBQfRB2aczpSX_GUyRr4TxE7bflpc8g

https://drive.google.com/uc?export=download&id=1VTUXanL66UL_YNN9twaWX1a0nspEhQb2


To download the trained model and the data run this code:

python download_data_trained_model.py



Downloading the preprocessed data from the google drive in pickle format as well as the pretrained model:

python download_data_trained_model.py


You can run the downloaded trained model on 3655 images which was obtained using the train2test split of 0.7.

The model was trained for 40 epochs on batch size of 2.

```bash

!python demo_test.py 

ConfusionMatrix = plt.imread('ConfusionMatrix.png')

plt.imshow(ConfusionMatrix)

```


You can train the model using the different arguments available in the utils/ClassifierOptions.py The following bash command will trigger a training for batch size 2: and 10 epochs. The trained model will be saved in the save_models directoy with the naming of "model_name+batch_size+number of classes+number of epochs.pt"


```bash

python demo_train.py --batch_size 2 --epochs 10  

```


And after training the model can be evaluated using following code:


```bash

python demo_test.py --saved_model_name /content/drive/MyDrive/BrainTumorCT2/saved_models/resnet50_2_2_10.pt

ConfusionMatrix = plt.imread('ConfusionMatrix.png')

plt.imshow(ConfusionMatrix)

```
