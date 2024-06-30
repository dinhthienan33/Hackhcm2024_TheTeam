from transformers import pipeline
from PIL import Image

pipe = pipeline("object-detection", model="facebook/detr-resnet-50")
img='BZ1A2269.jpg'
image=Image.open(img).convert("RGB")
result=pipe(image)[0]