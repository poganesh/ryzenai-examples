import torch
import timm
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from vit_utils import get_directories

# Parse command line arguments
parser = argparse.ArgumentParser(description='ViT Tiny Patch 16 on CIFAR-10')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
args = parser.parse_args()

# Download CIFAR-10 dataset and apply transformations
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# transform = transforms.Compose(
#         [transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()]
#     )

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# Load CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Load pretrained ViT model
model = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
num_classes = 10
model.head = nn.Linear(model.head.in_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(args.num_epochs):
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{args.num_epochs}, Loss: {running_loss/len(trainloader)}')

# Save trained model in PyTorch format
model.to("cpu")
torch.save(model.state_dict(), 'pretrained_vit.pth')
print("Model saved in PyTorch format.")

# Convert and save model in ONNX format
_, models_dir, data_dir, _ = get_directories()

dummy_input = torch.randn(1, 3, 224, 224)
input_names = ['input']
output_names = ['output']
dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
onnx_path = str(models_dir / 'pretrained_vit.onnx')
torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
print(f"Model saved in ONNX format: {onnx_path}")

print("done")
# Load model for testing
model.load_state_dict(torch.load(str(models_dir /'pretrained_vit.pth')))
model.to(device)
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')