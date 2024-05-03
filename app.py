from flask import Flask, render_template, request
import torch
from PIL import Image
from Python_files.models import CNN_Net
from Python_files.utility import class_map_dict
from torchvision import transforms


app = Flask(__name__)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
model = CNN_Net(in_channels=3,num_filter1=32)
model.load_state_dict(torch.load('cnn_best.pth',map_location=device))
transform = transforms.Compose([transforms.Resize(size=(32, 32)), transforms.ToTensor()])



@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    print(type(imagefile))
    image_path = "/test_Images"
    imagefile.save(image_path)
    test_image = Image.open(image_path)
    print(type(test_image))
    test_tensor = transform(test_image)
    logits = model(torch.unsqueeze(test_tensor, 0))
    probabilities = torch.softmax(logits, dim=1)
    max_prob, max_index = torch.max(probabilities, dim=1)
    classname = class_map_dict[max_index.item()]
    
    classification = '%s (%.2f%%)' % (classname, max_prob.item()*100)


    return render_template('index.html', prediction=classification)


if __name__ == '__main__':
    
    app.run(port=3000, debug=True)