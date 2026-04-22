from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def preprocess_image(path):
    img = Image.open(path).convert("L")
    return transform(img).unsqueeze(0)
