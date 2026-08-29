# Import modules.
import io
from pathlib import Path
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from .dataset import get_transforms
from .model import get_model

app = FastAPI(title='CIFAR-10 PyTorch Model Serving API')

# Save results of the model at intermediate stages.
CHECKPOINT_PATH = Path('/app/checkpoints/classifier_v1.pt')
if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH = Path('checkpoints/classifier_v1.pt')

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = None
transform = get_transforms(train=False)

@app.on_event('startup')
def load_model():
    '''Endpoint to load the Resnet18 model using FastAPI at /startup.'''
    global model
    if CHECKPOINT_PATH.exists():
        model = get_model(architecture='resnet18', num_classes=10, pretrained=False)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        model.to(device)
        model.eval()

@app.get('/health')
def health():
    '''Endpoint to check model status and error codes using /health.'''
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    return {'status': 'ok', 'model_loaded': True}

@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    '''Endpoint to serve images by the model in real time with /predict.'''
    if model is None:
        raise HTTPException(status_code=503, detail='Model standard not loaded')

    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
        tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1).squeeze(0)

        return {
            'prediction': CLASSES[torch.argmax(probs).item()],
            'probabilities': {CLASSES[i]: round(probs[i].item(), 4) for i in range(len(CLASSES))}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid image format: {str(e)}')