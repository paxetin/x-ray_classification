import torch
from torchvision import transforms
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def classify(file_path, model):
    transform = transforms.Compose([transforms.Resize((224, 224)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])
    
    img = Image.open(file_path, 'r').convert('RGB')
    input = transform(img).float().unsqueeze_(0).to(device)

    with torch.no_grad():
        model.eval()
        output = model(input)
        index = output.data.cpu().numpy().argmax()
        print(get_classes(index))

def get_classes(output):
    classes = {'Covid': 0, 'Lung_Opacity': 1, 'Normal': 2, 'Viral_Pneumonia': 3}
    return list(filter(lambda x: classes[x] == output, classes))[0]

if __name__ == '__main__':

    model = torch.load('model_4.pt')
    classify('./Data/Test/normal.jpg', model)