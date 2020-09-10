import torch
import torchvision
import torchvision.transforms as transforms


model  = torchvision.models.alexnet(pretrained=True)
for prarm in model.parameters():
    prarm.requires_grad = False
model.classifier[6] = torch.nn.Linear(4096, 10)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bir', 'cat', 'ddeer', 'dog', 'frog', 'horse', 'ship', 'truck')

