import matplotlib.pyplot as plt

# Accuracy values (replace if needed)
centralized_acc = 99.00
federated_iid_acc = 98.77
federated_noniid_acc = 35.26

methods = ["Centralized", "Federated IID", "Federated Non-IID"]
accuracies = [centralized_acc, federated_iid_acc, federated_noniid_acc]

plt.figure()
plt.bar(methods, accuracies)

plt.title("Accuracy Comparison: Centralized vs Federated Learning")
plt.ylabel("Accuracy (%)")
plt.xlabel("Training Method")

plt.savefig("accuracy_comparison.png")
plt.show()