import torch
import math
import torch.nn as nn
from torchsummary import summary


class SubModule(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob):
        super(SubModule, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.batchnorm = nn.BatchNorm1d(output_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        return x

class MySequentialModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, dropout_prob):
        super(MySequentialModel, self).__init__()
        self.layer1 = SubModule(input_size, hidden_size1, dropout_prob)
        self.layer2 = SubModule(hidden_size1, hidden_size2, dropout_prob)
        self.layer3 = SubModule(hidden_size2, hidden_size3, dropout_prob)
        self.output_layer = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        return x

class CNN_Net(nn.Module):
    def __init__(self,in_channels,num_filter1):
        super(CNN_Net,self).__init__()
        self.convblock1 = ConvBlock(in_channels,num_filter1)
        self.convblock2 = ConvBlock(num_filter1,num_filter1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.convblock3 = ConvBlock(num_filter1,2*num_filter1)
        self.convblock4 = ConvBlock(2*num_filter1,2*num_filter1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.convblock5 = ConvBlock(2*num_filter1,4*num_filter1)
        self.convblock6 = ConvBlock(4*num_filter1,4*num_filter1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4,4))
        self.linear1 = nn.Linear(128,64)
        self.linear2 = nn.Linear(64,10)

        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear,nn.Conv2d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self,x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.maxpool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.maxpool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.maxpool3(x)
        x = self.avgpool(x)
        flat_x = x.view(x.size(0),-1)
        x = self.linear1(flat_x)
        x = self.linear2(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=(3,3),padding=1)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=(3,3),padding=1)
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)
        return x



if __name__== '__main__':

    # Define the model parameters
    input_size = 3072  # Example input size
    hidden_size1, hidden_size2, hidden_size3 = 256, 128, 64  # Example hidden size
    output_size = 10  # Example output size
    dropout_prob = 0.15  # Example dropout probability

    # Create an instance of the model
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    dnn_model = MySequentialModel(input_size, hidden_size1,hidden_size2,hidden_size3, output_size, dropout_prob)
    print(summary(dnn_model.to(device), (input_size,), 1,'cuda'))

    # Define the model parameters
    in_channels = 3 # Example input size
    num_filter1 = 32
    # Create an instance of the model
    cnn_model = CNN_Net(in_channels,num_filter1)
    print(summary(cnn_model.to(device), (3,32,32), 1,'cuda'))


