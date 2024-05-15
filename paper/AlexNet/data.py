from torchvision import transforms
from torchvision import datasets
from torch.utils import data

def load_train_data():
    train_dataset = datasets.ImageFolder("./data/train", transforms.Compose([
        transforms.Resize((256, 256)), # 각 이미지의 사이즈가 달라 통일
        
        transforms.CenterCrop((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # ImageNet의 이미지들의 RGB 평균, 표준편차
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    return data.DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=128,
    )
