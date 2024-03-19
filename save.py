import torch
from unet import UNet

# Path to the pre-trained model
model_pth = r'G:\python\models\Single_SpineSegmentationv8_cpu.pth'

# Check if CUDA (GPU) is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the UNet model
model = UNet(in_channels=1, num_classes=1).to(device)

# Load the state dictionary of the pre-trained model
model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

# Create an example input tensor of size (1, 1, 128, 128)
example_input = torch.randn(1, 1, 128, 128).to(device)

# Trace the model with the example input
traced_model = torch.jit.trace(model, example_input)

# Save the traced model
torch.jit.save(traced_model, r'G:\python\models\Single_SpineSegmentationv8.2_cpu.pth')
