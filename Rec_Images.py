import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Modelo restnet preentrenado
model = models.resnet50(pretrained=True)
model.eval()

# Transformación de imagen para que puedan ser utilizadas en el modelo 
transform = transforms.Compose([
    transforms.Resize((224, 224)),#ajusta el tamaño de la imagen para que pueda ser utilizado en restnet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Función para predecir etiquetas
def predict(imagen_ruta, model):
    imagen = Image.open(imagen_ruta)
    input_tensor = transform(imagen).unsqueeze(0)  # transforma el tensor de 3D a 4D
    with torch.no_grad():#Desactiva el cálculo de gradientes para todas las operaciones dentro de su bloque. 
        output = model(input_tensor)
    _, prediccion = torch.max(output, 1)#Obtiene una etiqueta que tenga la mayor probabilidad
    return prediccion.item()

# Predicción por imagen especifica o por imagen individual
imagen_ruta = "/content/images/lugar1.jpg"##cambiar por: "ruta en donde se guardo el archivo"/images/"nombredelarchivo"
label = predict(imagen_ruta, model)
print(f"Numero de etiqueta: {label}")
imagen = Image.open(imagen_ruta)
plt.imshow(imagen)
plt.title(f"Numero de etiqueta: {label}")
plt.axis('off')
plt.show()



def prediccion_lote(lote, model):
  predictions = {}

  for filename in os.listdir(lote):
    if filename.endswith(".jpg"):
      image_path = os.path.join(lote, filename)
      label = predict(image_path, model)
      predictions[filename] = label
  return predictions

folder_path = "/content/images"##cambiar por: "ruta en donde se guardo el archivo"/images
results = prediccion_lote(folder_path, model)
print(results)


def visualize_predictions(folder_path, predictions):
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))  # Muestra 5 imágenes
    for i, (filename, label) in enumerate(list(predictions.items())[:5]):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        axes[i].imshow(image)
        axes[i].set_title(f"Etiqueta: {label}")
        axes[i].axis("off")
    plt.show()

visualize_predictions(folder_path, results)
