import torch
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {}".format(device))
model = torch.load("model_set/model_10", map_location=device)
model.eval()

z = torch.randn(1, 100, 1, 1, device=device)
new_music = model.forward(z)
save_image(new_music, "generate_image.png")
