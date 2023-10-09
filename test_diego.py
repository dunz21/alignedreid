import torch
import numpy as np
from torchvision import transforms as T

def load_model(model_path):
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'softmax', 'metric'}, aligned=True)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def extract_features(model, image_path):
    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform_test(image).unsqueeze(0)
    features, local_features = model(image)
    return features.data.cpu(), local_features.data.cpu()

model = load_model('checkpoint_ep300.pth.tar',map_location=torch.device('cpu'))
features, local_features = extract_features(model, 'path_to_your_image.jpg')
