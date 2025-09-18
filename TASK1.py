from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt

# Load CIFAR-10 from Hugging Face
dataset = load_dataset('cifar10')

# Define transforms: convert to tensor and normalize
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Custom Dataset wrapper to apply transforms
class CIFAR10TorchDataset(torch.utils.data.Dataset):
	def __init__(self, hf_dataset, transform=None):
		self.dataset = hf_dataset
		self.transform = transform
	def __len__(self):
		return len(self.dataset)
	def __getitem__(self, idx):
		img = self.dataset[idx]['img']
		label = self.dataset[idx]['label']
		if self.transform:
			img = self.transform(img)
		return img, label

# Prepare train and test datasets
train_dataset = CIFAR10TorchDataset(dataset['train'], transform=transform)
test_dataset = CIFAR10TorchDataset(dataset['test'], transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)




# Define a simple CNN architecture
class SimpleCNN(nn.Module):
	def __init__(self):
		super(SimpleCNN, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
		self.relu1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.relu2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(64 * 8 * 8, 128)
		self.relu3 = nn.ReLU()
		self.fc2 = nn.Linear(128, 10)
	def forward(self, x):
		x = self.pool1(self.relu1(self.conv1(x)))
		x = self.pool2(self.relu2(self.conv2(x)))
		x = x.view(x.size(0), -1)
		x = self.relu3(self.fc1(x))
		x = self.fc2(x)
		return x

# Instantiate model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 25
train_losses = []
for epoch in range(num_epochs):
	model.train()
	running_loss = 0.0
	for images, labels in train_loader:
		images, labels = images.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item() * images.size(0)
	epoch_loss = running_loss / len(train_loader.dataset)
	train_losses.append(epoch_loss)
	print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Plot training loss vs. epoch
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Epoch')
plt.grid(True)
plt.show()

# Model Evaluation
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
	for images, labels in test_loader:
		images, labels = images.to(device), labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs, 1)
		y_true.extend(labels.cpu().numpy())
		y_pred.extend(predicted.cpu().numpy())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print("\nPerformance Metrics Table:")
print(f"{'Metric':<12}{'Value':<10}")
print(f"{'Accuracy':<12}{accuracy:<10.4f}")
print(f"{'Precision':<12}{precision:<10.4f}")
print(f"{'Recall':<12}{recall:<10.4f}")
print(f"{'F1-Score':<12}{f1:<10.4f}")

#  Feature Map Extraction and Visualization
def visualize_feature_maps(model, image, device):
	model.eval()
	image = image.unsqueeze(0).to(device)
	with torch.no_grad():
		# Extract feature maps from conv1 and conv2
		x1 = model.relu1(model.conv1(image))
		x2 = model.relu2(model.conv2(model.pool1(x1)))
	# Plot feature maps for conv1
	plt.figure(figsize=(12, 4))
	for i in range(8):  # Show first 8 feature maps
		plt.subplot(2, 4, i+1)
		plt.imshow(x1[0, i].cpu().numpy(), cmap='viridis')
		plt.title(f'conv1 map {i+1}')
		plt.axis('off')
	plt.suptitle('Feature Maps from conv1')
	plt.show()
	# Plot feature maps for conv2
	plt.figure(figsize=(12, 4))
	for i in range(8):
		plt.subplot(2, 4, i+1)
		plt.imshow(x2[0, i].cpu().numpy(), cmap='viridis')
		plt.title(f'conv2 map {i+1}')
		plt.axis('off')
	plt.suptitle('Feature Maps from conv2')
	plt.show()

# Get a sample image from test set
sample_img, _ = test_dataset[0]
visualize_feature_maps(model, sample_img, device)

#  Hyperparameter Ablation Study
def run_experiment(lr, batch_size, num_filters, num_layers):
	# Dynamically build CNN with variable filters/layers
	class AblationCNN(nn.Module):
		def __init__(self):
			super(AblationCNN, self).__init__()
			layers = []
			in_channels = 3
			for i in range(num_layers):
				out_channels = num_filters
				layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
				layers.append(nn.ReLU())
				layers.append(nn.MaxPool2d(2, 2))
				in_channels = out_channels
			self.conv = nn.Sequential(*layers)
			self.fc1 = nn.Linear(num_filters * (32 // (2 ** num_layers)) * (32 // (2 ** num_layers)), 128)
			self.relu = nn.ReLU()
			self.fc2 = nn.Linear(128, 10)
		def forward(self, x):
			x = self.conv(x)
			x = x.view(x.size(0), -1)
			x = self.relu(self.fc1(x))
			x = self.fc2(x)
			return x
	# DataLoader with batch size
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
	model = AblationCNN().to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)
	# Train for fewer epochs for speed
	for epoch in range(5):
		model.train()
		for images, labels in train_loader:
			images, labels = images.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
	# Evaluate
	model.eval()
	y_true, y_pred = [], []
	with torch.no_grad():
		for images, labels in test_loader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs, 1)
			y_true.extend(labels.cpu().numpy())
			y_pred.extend(predicted.cpu().numpy())
	accuracy = accuracy_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred, average='macro')
	recall = recall_score(y_true, y_pred, average='macro')
	f1 = f1_score(y_true, y_pred, average='macro')
	return accuracy, precision, recall, f1

# Hyperparameter values
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]
num_filters_list = [16, 32, 64]
num_layers_list = [3, 5, 7]

results = []
for lr in learning_rates:
	for batch_size in batch_sizes:
		for num_filters in num_filters_list:
			for num_layers in num_layers_list:
				acc, prec, rec, f1 = run_experiment(lr, batch_size, num_filters, num_layers)
				results.append({
					'LR': lr,
					'Batch': batch_size,
					'Filters': num_filters,
					'Layers': num_layers,
					'Accuracy': acc,
					'Precision': prec,
					'Recall': rec,
					'F1-Score': f1
				})
				print(f"LR={lr}, Batch={batch_size}, Filters={num_filters}, Layers={num_layers} => Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

# Print summary table
print("\nTable 1: Performance Metrics Comparison of CNN Models")
print(f"{'LR':<6}{'Batch':<8}{'Filters':<8}{'Layers':<8}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}")
for r in results:
	print(f"{r['LR']:<6}{r['Batch']:<8}{r['Filters']:<8}{r['Layers']:<8}{r['Accuracy']:<10.4f}{r['Precision']:<10.4f}{r['Recall']:<10.4f}{r['F1-Score']:<10.4f}")