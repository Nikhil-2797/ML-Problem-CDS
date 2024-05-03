from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,datasets,models
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging

# Create a custom logger
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

log_tuple = namedtuple('log_tuple',['class_name','epoch_no','loss','accuracy'])

class_map_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
                  8: 'ship', 9: 'truck'}

Aug_List = [transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)]


def plot_curves(trainlog_dict, vallog_dict, acc_bool, path_str=None):
    ## data retrival
    train_data = trainlog_dict
    val_data = vallog_dict
    epochs = [num for num in train_data]
    if acc_bool:
        Allclass_train_acc = []
        Allclass_val_acc = []
        for i in range(11):
            Allclass_train_acc.append([tup.accuracy for value in train_data.values() for tup in value if tup.class_name == i])
            Allclass_val_acc.append([tup.accuracy for value in val_data.values() for tup in value if tup.class_name == i])
    else:
        Allclass_train_loss = []
        Allclass_val_loss = []
        for i in range(11):
            Allclass_train_loss.append([tup.loss for value in train_data.values() for tup in value if tup.class_name == i])
            Allclass_val_loss.append([tup.loss for value in val_data.values() for tup in value if tup.class_name == i])

    # Set up the figure and GridSpec layout
    fig = plt.figure(figsize=(17,10))
    # Create a gridspec for the first 10 subplots (2 rows, 5 columns)
    gs = gridspec.GridSpec(2, 5, figure=fig)
    # Generate the first 10 subplots
    for i in range(10):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        ax.set_title(f'Class {i+1} ({class_map_dict[i]})')
        ax.set_xlabel('Epochs')
        if acc_bool:
            ax.set_ylabel('Accuracy(%)')
            ax.plot(epochs, Allclass_train_acc[i],'r', label= 'Training plot')
            ax.plot(epochs ,Allclass_val_acc[i],'g',label= 'Validation plot')
            ax.legend()
            ax.set_xlim(0, len(epochs)+5)  # Set x-axis limits
            ax.set_ylim(40, 105)  # Set y-axis limits
        else:
            ax.set_ylabel('Loss')
            ax.plot(epochs, Allclass_train_loss[i],'r', label = 'Training plot')
            ax.plot(epochs ,Allclass_val_loss[i],'g', label = 'Validation plot')
            ax.legend()
            ax.set_xlim(0, len(epochs)+5)
            ax.set_ylim(40, 105)
    # Adjust the gridspec to make room for the 11th, larger subplot
    gs.update(bottom=0.4)  # You might need to adjust this value based on your actual figure sizes and desired layout

    # Add the 11th figure, manually adjusting its position and size
    # Here we place it in the middle of the third row, adjusting its width and height as necessary
    # The values for `left`, `bottom`, `width`, and `height` are fractions of the figure width and height
    ax11 = fig.add_axes([0.3, 0.1, 0.4, 0.2])  # left, bottom, width, height in figure coordinate
    ax11.set_title('Overall Performance on all classes')
    if acc_bool:
        ax11.set_ylabel('Accuracy(%)')
        ax11.plot(epochs, Allclass_train_acc[10],'r', label = 'Training plot')
        ax11.plot( epochs ,Allclass_val_acc[10],'g',label = 'Validation plot')
        ax11.legend()
        ax11.set_xlim(0, len(epochs)+5)
        ax11.set_ylim(40, 105)

    else:
        ax11.set_ylabel('Loss')
        ax11.plot(epochs, Allclass_train_loss[10],'r',label = 'Training plot')
        ax11.plot( epochs ,Allclass_val_loss[10],'g', label = 'Validation plot')
        ax11.legend()
        ax11.set_xlim(0, len(epochs)+5)
        ax11.set_ylim(40, 105)

    # Adjust subplot layout to avoid overlapping
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    if path_str is not None:
        plt.savefig(path_str +'.png', dpi=300, bbox_inches='tight')
    plt.show()





