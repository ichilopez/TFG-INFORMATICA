from abc import abstractmethod
import torch.nn as nn
class Model(nn.Module):

    from abc import abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

    def train(self, trainloader, epochs, learning_rate, device="cuda"):
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                               lr=learning_rate)
        for epoch in range(1, epochs + 1):
            self.train() 
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(trainloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step() #Se modifica su valor en memoria cuando llamamos a optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(trainloader)
            print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}")

    def evaluate(self, testloader, device="cuda"):
        self.eval() 
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad(): 
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(testloader)
        accuracy = correct / total
        print(f"Evaluation -> Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    def forward(self,x):
        return self.model(x)
 