import streamlit as st
from PIL import Image
import os
import warnings
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
import torch.nn.functional as F

warnings.filterwarnings('ignore')

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
def build_model():
    model = mobilenet_v2()
    model.classifier[-1] = nn.Linear(model.last_channel, len(classes))
    return model

folder_path = os.path.dirname(__file__)
model = build_model()
path = os.path.join(folder_path, 'model.pth')
model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.set_page_config('Trash Classifier', ':recycle:')
st.title('Trash Classifier :recycle:')
img_file = st.camera_input('Trash to Classify')

if img_file:
    img = Image.open(img_file)
    transformed_img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(transformed_img)
        probabilities = F.softmax(output, dim=1)
        _, predictions = torch.max(probabilities, 1)
        idx = predictions.item()
        label = classes[idx]
        conf = round(probabilities[0][idx].item(), 2)
        # print(label, conf)

        if conf > 0.6:
            st.info(f'AI Classifies this Trash as {label}')
        else:
            st.info('AI cannot detect/classify trash. Mark as Others.')
