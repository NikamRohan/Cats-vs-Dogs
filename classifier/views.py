import os, datetime, random
from django.conf import settings

'''Handling files'''
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage


from django.shortcuts import render

''' Pytorch Specific Libraries'''
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
from PIL import Image
from recognizer.settings import MEDIA_ROOT 

import os

def handle_uploaded_file(f):
    name = str(datetime.datetime.now().strftime('%H%M%S')) + str(random.randint(0, 1000)) + str(f)
    path = default_storage.save(MEDIA_ROOT + '/' + name,
                                ContentFile(f.read()))
    return os.path.join(MEDIA_ROOT, path), name


# Create your views here.
def home(request):
    if request.POST:
        img_classes = ['Cat','Dog']
        imgtovec = Img2Vec()
        file1_path, file1_name = handle_uploaded_file(request.FILES['file1'])
        value, image_one_class = imgtovec.image_loader(Image.open(file1_path))
        img_class = img_classes[image_one_class]



        return render(request,"classifier/base.html",{"Class" : img_class,"post":True,"img1src":file1_name})


    return render(request,"classifier/base.html",{'post':False})


class Img2Vec:
    def __init__(self, cuda=False):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.device = torch.device("cuda" if cuda else "cpu")
        #self.layer_output_size = layer_output_size
        self.model = models.vgg16(pretrained = True)

        #print(self.model)

        for param in self.model.parameters():
            param.requires_grad = False


        # for param in self.model.parameters():
        #     if param.requires_grad:
        #         print(param.shape)

        #print(self.model.classifier)

        self.model.classifier = nn.Sequential(
                  nn.Linear(25088,4096),
                  nn.ReLU(inplace=True),
                  nn.Dropout(p=0.2),
                  nn.Linear(4096,1000),
                  nn.ReLU(inplace=True),
                  nn.Dropout(p=0.2),
                  nn.Linear(1000,100),
                  nn.ReLU(inplace=True),
                  nn.Dropout(p=0.2),
                  nn.Linear(100,2),
        )


        #print(self.model)

        # for param in self.model.parameters():
        #     if param.requires_grad:
        #         print(param.shape)


        self.model = self.model.to(self.device)

        loss_fn = nn.CrossEntropyLoss()
        opt = optim.Adam(self.model.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        PATH = os.path.join(settings.BASE_DIR,"classifier/cats_dogs_trained_model.pt") 
        self.model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

        print("Loading Model......")
        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()


    def image_loader(self,img,tensor = False):
        image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)
        output = self.model(image)
        v,pred = torch.max(output.data,1)
        return v,pred

    





























  