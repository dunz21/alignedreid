
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import models
from util.distance import low_memory_local_dist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def remove_fc(state_dict):
    """Remove the fc layer parameters from state_dict."""
    new_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc')}
    return new_state_dict

def load_model(model_path, arch='resnet50'):
    model = models.init_model(name=arch, num_classes=0, loss={'softmax', 'metric'},aligned=True)
    checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
    model_dict = checkpoint['state_dict']
    pretrained_dict = {k: v for k, v in model_dict.items() if k not in ['classifier.weight', 'classifier.bias']}
    model.load_state_dict(pretrained_dict)
    model = model.cpu()
    model.eval()
    return model

# [1,3,256,128]
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    return img.unsqueeze(0)


def compute_distance(model, img_path1, img_path2, distance_type='global'):
    with torch.no_grad():
        img1 = preprocess_image(img_path1).cpu()
        img2 = preprocess_image(img_path2).cpu()

        features1, local_features1 = model(img1)
        features2, local_features2 = model(img2)

        features1 = features1 / (torch.norm(features1, p=2, dim=1, keepdim=True) + 1e-12)
        features2 = features2 / (torch.norm(features2, p=2, dim=1, keepdim=True) + 1e-12)

        global_distance = F.pairwise_distance(features1, features2, p=2).item()
    
        if distance_type == 'global':
            return global_distance
        # --unaligned',action= 'store_true', help= 'test local feature with unalignment'
        # Si esta el parametro, no va alinear, y va a tomar menos tiempo -> (Without DMLI)
        # Si no esta el parametro va a ser False, pero hay una negacion por ende va a permite DMLI
        local_distance_matrix = low_memory_local_dist(local_features1.cpu().numpy(), local_features2.cpu().numpy())
        local_distance = np.mean(local_distance_matrix)  # Placeholder: Use an appropriate method to reduce matrix to scalar


        if distance_type == 'local':
            return local_distance
        elif distance_type == 'global_local':
            return global_distance + local_distance
        else:
            raise ValueError("Invalid distance_type. Choose from 'global', 'local', or 'global_local'.")



def plot_pca(images, model, preprocess_image):
    """
    Plots the output of a model on images in 2D using PCA.
    
    Parameters:
    - images: List of images.
    - model: A model that will extract features.
    - preprocess_image: A function to preprocess images before passing to the model.
    """
    
    # Extract image names from paths
    image_names = [os.path.splitext(os.path.basename(img_path))[0] for img_path in images]
    
    # Extract features
    features_list = []
    for img in images:
        img = preprocess_image(img)
        with torch.no_grad():
            features, _ = model(img)
        features_list.append(features)
    
    # Concatenate features using torch.cat
    features_tensor = torch.cat(features_list, dim=0)
    
    # Convert tensor to numpy array
    features_array = features_tensor.cpu().numpy()
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_array)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(pca_result):
        plt.scatter(x, y, label=image_names[i])
    
    plt.legend()
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Model AlignedReId')
    plt.show()

if __name__ == "__main__":
    model_path = "Alignedreid_models/Cuhk03_Resnet50_Alignedreid/checkpoint_ep300.pth.tar" #FUNCIONA
    # model_path = "Alignedreid_models/Cuhk03_Resnet50_Alignedreid(LS)/checkpoint_ep300.pth.tar"
    # model_path = "Alignedreid_models/DukeMTMCReID_Resnet50_Alignedreid/checkpoint_ep300.pth.tar"
    # model_path = "Alignedreid_models/DukeMTMCReID_Resnet50_Alignedreid(LS)/checkpoint_ep300.pth.tar"
    # model_path = "Alignedreid_models/Market1501_Resnet50_Alignedreid/checkpoint_ep300.pth.tar"
    # model_path = "Alignedreid_models/Market1501_Resnet50_Alignedreid(LS)/checkpoint_ep300.pth.tar" #FUNCIOAN
    # model_path = "Alignedreid_models/MSMT17_Resnet50_Alignedreid/checkpoint_ep300.pth.tar"
    # model_path = "Alignedreid_models/MSMT17_Resnet50_Alignedreid(LS)/checkpoint_ep300.pth.tar"
    img1_path = "./people_2/img_3_20.png"
    img2_path = "./people_2/img_3_20.png"
    distance_type = 'local'  # Choose from 'global', 'local', or 'global_local'.

    model = load_model(model_path)
    # distance = compute_distance(model, img1_path, img2_path,distance_type)
    images_list = [
        './people_2/img_3_20.png',
        './people_2/img_3_580.png',
        './people_2/img_3_40.png',
        './people_2/img_3_60.png',
        './people_2/img_5_20.png',
        './people_2/img_5_40.png',
        './people_2/img_5_60.png',
        './people_2/img_5_80.png',
        './people_2/img_9_20.png',
        './people_2/img_9_40.png',
        './people_2/img_9_60.png',
        './people_2/img_9_80.png',
        ]
    plot_pca(images_list,model,preprocess_image)
    # print("Squared Euclidean Distance between images:", distance)


