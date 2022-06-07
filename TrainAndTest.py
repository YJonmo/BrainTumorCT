import os
import time
import torch
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as  nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, models
from utils.data_set import BrainTumorDataset
from sklearn.metrics import confusion_matrix, classification_report

class TrainerAndTester:
    def __init__(self, Vars):
        self.vars = Vars
        self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)

    def save_model(self, state, is_best):
        filename=os.path.join(self.vars.ROOT, self.vars.save_path, self.vars.model + '_' + self.vars.batch_size + '_' + self.vars.epochs + '.pth.tar')
        torch.save(state, filename)

    def get_data(self):
        train_data = pickle.load(open(self.vars.pickle_path_train, 'rb'))
        test_data = pickle.load(open(self.vars.pickle_path_test, 'rb'))
        Xt, Yt, label_X, label_Y, IDs_X, IDs_Y= [], [], [], [], [], []
        ID, label, image = None, None, None

        for ID, label, image in train_data:
            IDs_X.append(ID)
            label_X.append(label)
            Xt.append(image)

        ID, label, image = None, None, None
        for ID, label, image in test_data:
            IDs_Y.append(ID)
            label_Y.append(label)
            Yt.append(image)

        train_set = BrainTumorDataset(Xt, label_X, self.vars.no_classes)
        test_set = BrainTumorDataset(Yt, label_Y, self.vars.no_classes)
        #val_set = BrainTumorDataset(Yt, label_Y, self.vars.no_classes)

        return train_set, test_set

    def get_model(self):

        if self.vars.model == 'resnet50':
            trainingModel = models.resnet50(pretrained=True)
        elif self.vars.model == 'resnet152':
            trainingModel = models.resnet152(pretrained=True)
        elif self.vars.model == 'resnet18':
            trainingModel = models.resnet18(pretrained=True)

        # Training all parameters
        for param in trainingModel.parameters():
            param.requires_grad = True
        # Input for fully connected layer
        n_inputs = trainingModel.fc.in_features

        trainingModel.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                         nn.SELU(),
                                         nn.Dropout(p=0.4),
                                         nn.Linear(2048, 2048),
                                         nn.SELU(),
                                         nn.Dropout(p=0.4),
                                         nn.Linear(2048, self.vars.no_classes),
                                         nn.LogSigmoid())

        for name, child in trainingModel.named_children():
            for name2, params in child.named_parameters():
                params.requires_grad = True

        # Define the loss and optimizer
        self.crossen_loss = nn.CrossEntropyLoss().to(self.device)
        self.sgd_opt = torch.optim.SGD(trainingModel.parameters(), momentum=0.9, lr=3e-4)

        return trainingModel


    def train(self):
        train_set, test_set = self.get_data()

        train_loader = DataLoader(train_set, batch_size=self.vars.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=2)
        # val_loader = DataLoader(val_set, batch_size=self.vars.batch_size, shuffle=True, pin_memory=True, num_workers=1)
        test_loader = DataLoader(test_set, batch_size=self.vars.batch_size, shuffle=True, pin_memory=True,
                                 num_workers=2)


        trainingModel = self.get_model()
        trainingModel.to(self.device)

        # Lists to store train, test losses and accuracies
        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        best_prec_loss = 2

        # Training loop
        start_time = time.time()
        # Batch Variables
        b = None
        train_b = None
        test_b = None

        # Training
        for i in range(self.vars.epochs):
            # Training Correct and Validation Correct is updated to 0 during every iteration
            train_corr = 0
            val_corr = 0
            # Timing Epoch: starting epoch time
            e_start = time.time()
            # Batch Training
            for b, (Y, X) in enumerate(train_loader):
                X, Y = X.to(self.device), Y.to(self.device)
                # Forward passing image
                Y_pred = trainingModel(X.view(-1, 3, self.vars.image_size, self.vars.image_size))
                # Loss calculation
                loss = self.crossen_loss(Y_pred.float(), torch.argmax(Y.view(-1, self.vars.no_classes), dim=1).long())
                # The argmax of the predicted tensor is assigned as our label
                mod_pred = torch.argmax(Y_pred, dim=1).data
                # # of correctly predicted images in a batch is equal to the sum of all correctly predicted images
                batch_corr = (mod_pred == torch.argmax(Y.view(-1, self.vars.no_classes), dim=1)).sum()
                # Batch correct is added to train correct for tracking the # of correctly predicted labels of the training data
                train_corr += batch_corr
                # Setting SGD opt gradient to zero
                self.sgd_opt.zero_grad()
                # Back propagation based on the loss to update weights to improve learning
                loss.backward()
                # Performing the increment in weights by taking a step
                self.sgd_opt.step()
            # Epoch end time
            e_end = time.time()
            # Output: Training Metrics
            print(f'Current Epoch: {(i + 1)} Running Batch: {(b + 1) * self.vars.batch_size} \nTraining Accuracy: '
                  f'{train_corr.item() * 100 / (self.vars.batch_size * b):2.2f} %  Training Loss: {loss.item():2.6f}  '
                  f'Training Duration: {((e_end - e_start) / 60):.2f} minutes')
            # Storing the training losses and correct to plot graph between losses and predicted corrects with batch
            train_b = b
            train_losses.append(loss)
            train_accs.append(train_corr)

            X, y = None, None

            # Validation using val_loader data
            # The backpropagation isn't performed as it is validation data
            with torch.no_grad():
                for b, (Y, X) in enumerate(test_loader):
                    # set label as cuda if device is cuda
                    X, Y = X.to(device), y.to(device)

                    # forward pass image
                    Y_val = trainingModel(X.view(-1, 3, self.vars.image_size, self.vars.image_size))

                    # The argmax of the predicted tensor is assigned as our label
                    mod_pred = torch.argmax(Y_val, dim=1).data
                    mod_pred[0]
                    # Batch correct is added to validation correct for tracking the # of correctly predicted labels of the validation data
                    val_corr += (mod_pred == torch.argmax(y.view(-1, self.vars.no_classes), dim=1)).sum()

            # Loss of validation set
            loss = self.crossen_loss(Y_val.float(), torch.argmax(y.view(-1, self.vars.no_classes), dim=1).long())
            # Output validation metrics
            print(
                f'Validation Accuracy {val_corr.item() * 100 / (self.vars.batch_size * b):2.2f} Validation Loss: {loss.item():2.6f}\n')

            # Saves model if the current validation loss less than the previous validation loss
            best = loss < best_prec_loss
            best_prec_loss = min(loss, best_prec_loss)
            self.save_model({
                'epoch': i + 1,
                'state_dict': trainingModel.state_dict(),
                'best_prec1': best_prec_loss,
            }, best)

            # Storing the validation losses and correct to plot graph between losses and predicted corrects with batch
            val_b = b
            val_losses.append(loss)
            val_accs.append(val_corr)

        # Training process's end time
        end_time = time.time() - start_time

        # Total training duration
        print("\nTraining Duration {:.2f} minutes".format(end_time / 60))

        #saving the model
        torch.cuda.empty_cache()
        torch.save(trainingModel.state_dict(), os.path.join(self.vars.ROOT, self.vars.model +
                str(self.vars.batch_size) + '_' + str(self.vars.no_classes) + '_' + str(self.vars.epochs) + '.pt'))

        Y_val = trainingModel(X.view(-1, 3, self.vars.image_size, self.vars.image_size))
        # The argmax of the predicted tensor is assigned as our label
        mod_pred = torch.argmax(Y_val, dim=1).data
        mod_pred[0]
        # Batch correct is added to validation correct for tracking the # of correctly predicted labels of the validation data
        val_corr += (mod_pred == torch.argmax(Y.view(-1, self.vars.no_classes), dim=1)).sum()
        loss = self.crossen_loss(Y_val.float(), torch.argmax(Y.view(-1, self.vars.no_classes), dim=1).long())
        print(
            f'Validation Accuracy {val_corr.item() * 100 / (self.vars.batch_size * b):2.2f} Validation Loss: {loss.item():2.6f}\n')


        # plot the training and validation losses
        train_losses_cpu = np.zeros(self.vars.epochs, dtype=np.float32)
        val_losses_cpu = np.zeros(self.vars.epochs, dtype=np.float32)
        for i in range(len(train_losses)):
            train_losses_cpu[i] = train_losses[i].cpu().data.numpy()
            val_losses_cpu[i] = val_losses[i].cpu().data.numpy()
        plt.plot(train_losses_cpu, label='Training loss')
        plt.plot(val_losses_cpu, label='Validation loss')
        plt.title('Loss Metrics')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()

        #plot the accuracy of the training and validation
        train_accs_cpu = np.zeros(self.vars.epochs, dtype=np.float32)
        val_accs_cpu = np.zeros(self.vars.epochs, dtype=np.float32)
        for i in range(len(train_accs_cpu)):
            train_accs_cpu[i] = train_accs[i].data.cpu().numpy()
            val_accs_cpu[i] = val_accs[i].cpu().data.numpy()
        plt.plot([t / 171 for t in train_accs_cpu], label='Training accuracy')
        plt.plot([t / 36 for t in val_accs_cpu], label='Validation accuracy')
        plt.title('Accuracy Metrics')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()


    def test(self, model_path):
        train_set, test_set = self.get_data()
        test_loader = DataLoader(test_set, 1, shuffle=True, pin_memory=True, num_workers=1)

        device = torch.device(self.device_name)
        trainingModel = self.get_model()
        trainingModel.to(device)
        # trainingModel.load_state_dict(torch.load('/content/drive/MyDrive/BrainTumorCT/bt_resnet50_bt_intermodel.pth.tar'))
        # filename = os.path.join(self.vars.ROOT, self.vars.save_path, self.vars.model + '_' + self.vars.batch_size + '_' + self.vars.epochs + '.pth.tar')
        #Stats = torch.load(filename)
        #trainingModel.dict = (Stats['state_dict'])
        trainingModel.load_state_dict(torch.load(model_path))
        train_loader, val_loader, train_set, val_set = None, None, None, None

        # Setting model to evaluation mode
        trainingModel.eval()

        # No weight updates
        with torch.no_grad():
            # some metrics storage for visualization and analysis
            test_pred = 0
            test_loss, test_corr, labels, pred = [], [], [], []

            # Batch wise test set evaluation
            for (y, X) in test_loader:
                # Use GPU if available
                X, y = X.to(device), y.to(device)
                print('X.shape')
                print(X.shape)
                print('y.shape')
                print(y.shape)

                # Original labels
                #labels.append(torch.argmax(y.view(self.vars.batch_size, self.vars.no_classes), dim=1).data)
                #labels.append(torch.argmax(y.view(1, self.vars.no_classes), dim=1).data)
                labels.append(torch.argmax(y.view(1, self.vars.no_classes), dim=1).data)

                print('labels.shape')
                print(labels[0].shape)
                # Forward pass
                y_val = trainingModel(X.view(-1, 3, self.vars.image_size, self.vars.image_size))
                print('y_val.shape')
                print(y_val.shape)
                print('X.view(-1, 3, self.vars.image_size,self.vars.image_size)')
                print(X.view(-1, 3, self.vars.image_size, self.vars.image_size).shape)
                # The argmax of the predicted tensor is assigned as our label
                pred_label = torch.argmax(y_val, dim=1).data
                print('pred_label.shape')
                print(pred_label.shape)
                # Predicted label addition to the list
                pred.append(pred_label)
                print('pred.shape')
                print(pred[0].shape)
                # Compute loss
                #loss = self.crossen_loss(y_val.float(),
                #                    torch.argmax(y.view(self.vars.batch_size, self.vars.no_classes), dim=1).long())
                loss = self.crossen_loss(y_val.float(), torch.argmax(y.view(1, self.vars.no_classes), dim=1).long())
                # Adding the total correct predicted per batch to overall correct predictions
                test_pred += (pred_label == torch.argmax(y.view(1, self.vars.no_classes), dim=1)).sum()

                # Storing correct
                test_corr.append(test_pred)
                test_loss.append(loss)

        print(f"Test Loss: {test_loss[-1].item():.4f}")

        labels = torch.stack(labels)
        pred = torch.stack(pred)

        #plot the confusion matrix
        LABELS = ['Healthy', 'Tumor']
        conf_matrix = confusion_matrix(pred.view(-1).cpu(), labels.view(-1).cpu())
        df_cm = pd.DataFrame(conf_matrix, LABELS, LABELS)
        plt.figure(figsize=(9, 6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='YlGnBu')
        plt.xlabel("Prediction")
        plt.ylabel("Target")
        plt.show()
