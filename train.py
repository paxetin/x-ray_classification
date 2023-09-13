import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from models.simple_model import CNN
from models.vgg import VGG
from models.resnet import *
from utils import *
import gc
from loguru import logger


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    path = './Data'
    batch_size = 24
    num_epochs = 10

    transform = transforms.Compose([transforms.Resize((224, 224)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])

    train_data = ImageFolder(f'{path}/Train', transform=transform)
    test_data = ImageFolder(f'{path}/Val', transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # model = CNN(num_classes=len(train_data.classes)).to(device)
    model = VGG('VGG19').to(device)
    # model = ResNet34().to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-5)
    summary = Summary(train_loader, test_loader)

    print(train_data.class_to_idx)

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            summary.compute(batch_idx, loss, outputs, labels, 'train')

        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                summary.compute(batch_idx, loss, outputs, labels, 'val')

        logger.info(f'∥ Epoch {form_str(epoch+1, num_epochs)} / {num_epochs} | train loss: {summary.train_loss:.4f} | train acc: {summary.train_acc:.4f} | val loss: {summary.val_loss:.4f} | val acc: {summary.val_acc:.4f} ∥')
        gc.collect()

    summary.visualize()
    torch.save(model, f'model_{len(train_data.classes)}_classes.pt')


