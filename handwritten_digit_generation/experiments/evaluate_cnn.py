# handwritten_digit_generation/experiments/evaluate_cnn.py

import os
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from handwritten_digit_generation.models.cnn import CNN
from handwritten_digit_generation.utils.file_utils import save_to_results, get_project_root
from handwritten_digit_generation.utils.visualization import plot_confusion_matrix

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def evaluate_cnn():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root=os.path.join(project_root, 'data'), train=False, download=True,
                                  transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = CNN().to(device)
    model_path = save_to_results('cnn_model.pth', subdirectory='cnn')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    accuracy = 100 * correct / total
    print(f'Overall accuracy: {accuracy:.2f}%')

    for i in range(10):
        class_accuracy = 100 * class_correct[i] / class_total[i]
        print(f'Accuracy of {i}: {class_accuracy:.2f}%')

    # Draw confusion matrix
    cm_path = save_to_results('cnn_confusion_matrix.png', subdirectory='cnn')
    plot_confusion_matrix(all_labels, all_preds, cm_path)

    result_path = save_to_results('cnn_evaluation_result.txt', subdirectory='cnn')
    with open(result_path, 'w') as f:
        f.write(f'Overall accuracy: {accuracy:.2f}%\n')
        for i in range(10):
            class_accuracy = 100 * class_correct[i] / class_total[i]
            f.write(f'Accuracy of {i}: {class_accuracy:.2f}%\n')

    return accuracy, class_correct, class_total


if __name__ == "__main__":
    evaluate_cnn()