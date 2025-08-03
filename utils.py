import matplotlib.pyplot as plt

def plot_metric(losses, accs, title="Training Metrics Over Epochs"):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Loss', marker='o')
    plt.plot(accs, label='Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_metrics(lrs, title="Learning_Rate"):
    plt.figure(figsize=(10, 6))
    plt.plot(lrs)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
    
def plot_metrics(values, title="Metric"):
    plt.figure(figsize=(10, 6))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(title)  # 這裡直接用 title 當 y 標籤
    plt.grid(True)
    plt.show()