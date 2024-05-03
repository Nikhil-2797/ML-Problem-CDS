import torch
import pickle
import copy
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from utility import log_tuple , Aug_List , log
from datasets import download_dataset,Cifar10_cnn,Cifar10_dnn,DataLoader
from models import MySequentialModel, CNN_Net

MATRIX_LOSS_NDX = 0
MATRIX_LABEL_NDX = 1
MATRIX_PRED_NDX = 2
MATRIX_SIZE = 3

# Download the dataset

(tr_images, tr_targets),(val_images,val_targets) = download_dataset()


class Training_app_dnn:
    def __init__(self,epochs,batch_size,bool_data_aug=False, bool_l2_reg=False, hidden_layer_nodes=[256, 128, 64], Dropout_prob = 0.20):
        self.use_device = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_device else 'cpu')
        self.epochs = epochs
        self.batch_size = batch_size
        self.node_lst = hidden_layer_nodes
        self.dropout = Dropout_prob
        self.l2_Reg = bool_l2_reg
        self.bool_data_aug = bool_data_aug
        self.data_aug = Aug_List
        self.model = self.initModel()
        self.optim = self.initOptim()
        self.trainingSampleCount = 0
        self.trainlogs = {}
        self.vallogs = {}

    def initModel(self):
        model = MySequentialModel(3072, self.node_lst[0], self.node_lst[1], self.node_lst[2], 10, self.dropout)
        if self.use_device:
            model = model.to(self.device)
        return model

    def initOptim(self):
        if self.l2_Reg:
            optimizer = Adam(self.model.parameters(), lr = 10e-3, weight_decay=0.005)
        else:
            optimizer = Adam(self.model.parameters(), lr = 10e-3)
        return optimizer

    def initTraindl(self):
        if self.bool_data_aug:
            train_dataset = Cifar10_dnn(tr_images,tr_targets,aug=self.data_aug)
        else:
            train_dataset = Cifar10_dnn(tr_images,tr_targets)
        train_dl = DataLoader(train_dataset,self.batch_size,shuffle=True,pin_memory=self.use_device)
        return train_dl

    def initValdl(self):
        val_dataset = Cifar10_dnn(val_images,val_targets)
        val_dl = DataLoader(val_dataset,self.batch_size,shuffle=False,pin_memory=self.use_device)
        return val_dl

    def main(self):
        train_dl = self.initTraindl()
        val_dl = self.initValdl()

        for i in range(1, self.epochs+1):
            epoch_report_train = self.doTraining(train_dl)
            self.logMatrices(epoch_report_train,i,True)
            epoch_report_val = self.doValidation(val_dl)
            self.logMatrices(epoch_report_val,i,False)

    def doTraining(self,train_dl):
        self.model.train()
        trainMatrix = torch.zeros(MATRIX_SIZE,len(train_dl.dataset),device = self.device)

        for batch_ndx , batch_data in enumerate(train_dl):
            self.optim.zero_grad()
            batch_loss = self.computeBatchLoss(batch_ndx , batch_data, train_dl.batch_size,trainMatrix)

            batch_loss.backward()

            self.optim.step()

        self.trainingSampleCount += len(train_dl.dataset)
        return trainMatrix.to('cpu')

    def doValidation(self,val_dl):
        with torch.no_grad():
            self.model.eval()
            valMatrix = torch.zeros(MATRIX_SIZE,len(val_dl.dataset),device = self.device)

            for batch_ndx ,batch_data in enumerate(val_dl):
                self.computeBatchLoss(batch_ndx,batch_data,val_dl.batch_size,valMatrix)
        return valMatrix.to('cpu')

    def computeBatchLoss(self,batch_ndx,batch_data,batch_size,log_matric):
        input_t, label_t = batch_data
        input_g = input_t.to(self.device,non_blocking = True)
        label_g = label_t.to(self.device,non_blocking = True)

        logits  = self.model(input_g)
        class_prob = torch.nn.Softmax(dim=1)(logits)
        pred = torch.argmax(class_prob,dim=1)
        loss_fn = CrossEntropyLoss(reduction='none')
        batch_loss = loss_fn(logits,label_g)

        start_ndx = batch_ndx*batch_size
        end_ndx = start_ndx + label_t.size(0)
        log_matric[MATRIX_LOSS_NDX,start_ndx:end_ndx] = batch_loss.detach()
        log_matric[MATRIX_LABEL_NDX,start_ndx:end_ndx] = label_g.detach()
        log_matric[MATRIX_PRED_NDX,start_ndx:end_ndx] = pred.detach()

        return batch_loss.mean()

    def logMatrices(self,epoch_record,epoch_no,train_bool):
        trainlogs=[]
        vallogs=[]
        mask_dict_labels = {i:epoch_record[MATRIX_LABEL_NDX] == i for i in range(0,10)}
        mask_dict_pred = {i:epoch_record[MATRIX_PRED_NDX] == i for i in range(0,10)}
        ### Loss calculation
        loss_per_class = [epoch_record[MATRIX_LOSS_NDX,mask_dict_labels[i]].mean() for i in range(0,10)]
        total_loss = epoch_record[MATRIX_LOSS_NDX].mean()
        loss_per_class.append(total_loss)
        ### Accuracy calculation
        accuracy_per_class = [(int((mask_dict_labels[i] & mask_dict_pred[i]).sum())/int(mask_dict_labels[i].sum()))*100 for i in range(0,10)]
        total_accuracy = (int((epoch_record[MATRIX_LABEL_NDX] == epoch_record[MATRIX_PRED_NDX]).sum())/int((epoch_record[MATRIX_LABEL_NDX] == epoch_record[MATRIX_LABEL_NDX]).sum()))*100
        accuracy_per_class.append(total_accuracy)
        for i in range(0,11):
            if train_bool:
                trainlogs.append(log_tuple(i,epoch_no,loss_per_class[i],accuracy_per_class[i]))
            else:
                vallogs.append(log_tuple(i,epoch_no,loss_per_class[i],accuracy_per_class[i]))

        if train_bool:
            self.trainlogs[epoch_no] = trainlogs
            log.info(f'Epoch:{epoch_no} , Total Training Loss: {total_loss} and Total training Accuracy is {total_accuracy} %')
        else:
            self.vallogs[epoch_no] = vallogs
            log.info(f'Epoch:{epoch_no} , Total Validation Loss: {total_loss} and Total Validation Accuracy is {total_accuracy} %')


class Training_app_cnn:
    def __init__(self,epochs,batch_size,in_channels = 3,num_filter1 = 32,bool_data_aug=False, bool_l2_reg=False):
        self.use_device = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_device else 'cpu')
        self.model_parameters = [in_channels,num_filter1]
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_Reg = bool_l2_reg
        self.data_aug = Aug_List
        self.bool_data_aug = bool_data_aug
        self.model = self.initModel()
        self.optim = self.initOptim()
        self.trainingSampleCount = 0
        self.trainlogs = {}
        self.vallogs = {}

    def initModel(self):
        model = CNN_Net(self.model_parameters[0],self.model_parameters[1])
        if self.use_device:
            model = model.to(self.device)
        return model

    def initOptim(self):
        if self.l2_Reg:
            optimizer = Adam(self.model.parameters(), lr = 10e-4, weight_decay=0.001)
        else:
            optimizer = Adam(self.model.parameters(), lr = 10e-4)
        return optimizer

    def initTraindl(self):
        if self.bool_data_aug:
            train_dataset = Cifar10_cnn(tr_images,tr_targets,aug=self.data_aug)
        else:
            train_dataset = Cifar10_cnn(tr_images,tr_targets)
        train_dl = DataLoader(train_dataset,self.batch_size,shuffle=True,pin_memory=self.use_device)
        return train_dl

    def initValdl(self):
        val_dataset = Cifar10_cnn(val_images,val_targets)
        val_dl = DataLoader(val_dataset,self.batch_size,shuffle=False,pin_memory=self.use_device)
        return val_dl

    def main(self):
        train_dl = self.initTraindl()
        val_dl = self.initValdl()

        for i in range(1, self.epochs+1):
            epoch_report_train = self.doTraining(train_dl)
            self.logMatrices(epoch_report_train,i,True)
            epoch_report_val = self.doValidation(val_dl)
            self.logMatrices(epoch_report_val,i,False)

    def doTraining(self,train_dl):
        self.model.train()
        trainMatrix = torch.zeros(MATRIX_SIZE,len(train_dl.dataset),device = self.device)

        for batch_ndx , batch_data in enumerate(train_dl):
            self.optim.zero_grad()
            batch_loss = self.computeBatchLoss(batch_ndx , batch_data, train_dl.batch_size,trainMatrix)

            batch_loss.backward()

            self.optim.step()

        self.trainingSampleCount += len(train_dl.dataset)
        return trainMatrix.to('cpu')

    def doValidation(self,val_dl):
        with torch.no_grad():
            self.model.eval()
            valMatrix = torch.zeros(MATRIX_SIZE,len(val_dl.dataset),device = self.device)

            for batch_ndx ,batch_data in enumerate(val_dl):
                self.computeBatchLoss(batch_ndx,batch_data,val_dl.batch_size,valMatrix)
        return valMatrix.to('cpu')

    def computeBatchLoss(self,batch_ndx,batch_data,batch_size,log_matric):
        input_t, label_t = batch_data
        input_g = input_t.to(self.device,non_blocking = True)
        label_g = label_t.to(self.device,non_blocking = True)

        logits  = self.model(input_g)
        class_prob = torch.nn.Softmax(dim=1)(logits)
        pred = torch.argmax(class_prob,dim=1)
        loss_fn = CrossEntropyLoss(reduction='none')
        batch_loss = loss_fn(logits,label_g)

        start_ndx = batch_ndx*batch_size
        end_ndx = start_ndx + label_t.size(0)
        log_matric[MATRIX_LOSS_NDX,start_ndx:end_ndx] = batch_loss.detach()
        log_matric[MATRIX_LABEL_NDX,start_ndx:end_ndx] = label_g.detach()
        log_matric[MATRIX_PRED_NDX,start_ndx:end_ndx] = pred.detach()

        return batch_loss.mean()

    def logMatrices(self,epoch_record,epoch_no,train_bool):
        trainlogs=[]
        vallogs=[]
        mask_dict_labels = {i:epoch_record[MATRIX_LABEL_NDX] == i for i in range(0,10)}
        mask_dict_pred = {i:epoch_record[MATRIX_PRED_NDX] == i for i in range(0,10)}
        ### Loss calculation
        loss_per_class = [epoch_record[MATRIX_LOSS_NDX,mask_dict_labels[i]].mean() for i in range(0,10)]
        total_loss = epoch_record[MATRIX_LOSS_NDX].mean()
        loss_per_class.append(total_loss)
        ### Accuracy calculation
        accuracy_per_class = [(int((mask_dict_labels[i] & mask_dict_pred[i]).sum())/int(mask_dict_labels[i].sum()))*100 for i in range(0,10)]
        total_accuracy = (int((epoch_record[MATRIX_LABEL_NDX] == epoch_record[MATRIX_PRED_NDX]).sum())/int((epoch_record[MATRIX_LABEL_NDX] == epoch_record[MATRIX_LABEL_NDX]).sum()))*100
        accuracy_per_class.append(total_accuracy)
        for i in range(0,11):
            if train_bool:
                trainlogs.append(log_tuple(i,epoch_no,loss_per_class[i],accuracy_per_class[i]))
            else:
                vallogs.append(log_tuple(i,epoch_no,loss_per_class[i],accuracy_per_class[i]))

        if train_bool:
            self.trainlogs[epoch_no] = trainlogs
            log.info(f'Epoch:{epoch_no} , Total Training Loss: {total_loss} and Total training Accuracy is {total_accuracy} %')
        else:
            self.vallogs[epoch_no] = vallogs
            log.info(f'Epoch:{epoch_no} , Total Validation Loss: {total_loss} and Total Validation Accuracy is {total_accuracy} %')


def do_validation(reloded_model,batch_size):
    val_dataset = Cifar10_cnn(val_images,val_targets)
    use_device = torch.cuda.is_available()
    device = torch.device('cuda' if use_device else 'cpu')
    val_dl = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,pin_memory=use_device)
    reloded_model = reloded_model.to(device)
    with torch.no_grad():
        reloded_model.eval()
        valMatrix = torch.zeros(MATRIX_SIZE,len(val_dl.dataset),device = device)
        for batch_ndx ,batch_data in enumerate(val_dl):
            input_t, label_t = batch_data
            input_g = input_t.to(device,non_blocking = True)
            label_g = label_t.to(device,non_blocking = True)
            logits  = reloded_model(input_g)
            class_prob = torch.nn.Softmax(dim=1)(logits)
            pred = torch.argmax(class_prob,dim=1)

            start_ndx = batch_ndx*batch_size
            end_ndx = start_ndx + label_t.size(0)
            valMatrix[MATRIX_LABEL_NDX,start_ndx:end_ndx] = label_g.detach()
            valMatrix[MATRIX_PRED_NDX,start_ndx:end_ndx] = pred.detach()

    epoch_record = valMatrix.to('cpu')
    mask_dict_labels = {i:epoch_record[MATRIX_LABEL_NDX] == i for i in range(0,10)}
    mask_dict_pred = {i:epoch_record[MATRIX_PRED_NDX] == i for i in range(0,10)}
    ### Accuracy calculation
    accuracy_per_class = [(int((mask_dict_labels[i] & mask_dict_pred[i]).sum())/int(mask_dict_labels[i].sum()))*100 for i in range(0,10)]
    total_accuracy = (int((epoch_record[MATRIX_LABEL_NDX] == epoch_record[MATRIX_PRED_NDX]).sum())/int((epoch_record[MATRIX_LABEL_NDX] == epoch_record[MATRIX_LABEL_NDX]).sum()))*100
    accuracy_per_class.append(total_accuracy)
    print(accuracy_per_class)



def save_model_logs(model_class,save_pth,list_str):

    torch.save(model_class.model.state_dict(), save_pth + f'\{list_str[0]}'+'.pth')

    train_logs = copy.deepcopy(model_class.trainlogs)
    val_logs = copy.deepcopy(model_class.vallogs)

    # Save the dictionary object to a file using pickle
    with open(save_pth + f'\{list_str[1]}' + '.pkl', 'wb') as f:
        pickle.dump(train_logs, f)

    with open(save_pth + f'\{list_str[2]}' + '.pkl', 'wb') as f:
        pickle.dump(val_logs, f)



def main(model_type,batch_size,epochs,save_dir,list_str=['last_model','train','val'],bool_l2_reg=False,bool_data_aug=False):
    if model_type.lower() =='dnn':
        Model_training_class = Training_app_dnn(epochs=epochs,batch_size=batch_size,bool_l2_reg=bool_l2_reg,bool_data_aug=bool_data_aug)
        Model_training_class.main()
    if model_type.lower() =='cnn':
        Model_training_class = Training_app_cnn(epochs=epochs,batch_size=batch_size,bool_l2_reg=bool_l2_reg,bool_data_aug=bool_data_aug)
        Model_training_class.main()

    save_model_logs(Model_training_class,save_dir,list_str=list_str)



if __name__ == '__main__':
    main('cnn',batch_size=256,epochs=100,save_dir='E:\Projects for CDS\ML-Problem-CDS\model_saved',bool_l2_reg=True,bool_data_aug=True)

