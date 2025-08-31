#python MIXER/main.py

# 模組載入
from train import run_gga, train_with_config
import matplotlib.pyplot as plt
import os
import jax

# 自動選擇 GPU 或 CPU
if any(device.platform == "gpu" for device in jax.devices()):
    os.environ["JAX_PLATFORM_NAME"] = "gpu"
    print("✅ 使用 GPU 執行")
else:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    print("⚠️ 未偵測到 GPU，自動切換為 CPU 執行")

# 類別名稱
dataset_name = "mnist"  # 直接訓練 mnist
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def main():
    mode = "train"  # 直接訓練
    optimizer = "adamw"
    earlystop = "n"
    num_epochs = 30
    pop_size = 10
    generations = 10
    batch_size = 128

    if mode == "train":
        default_config = {
            "num_blocks": 2,
            "patch_size": 4,
            "hidden_dim": 128,
            "tokens_mlp_dim": 64,
            "channels_mlp_dim": 512,
            "dropout_rate": 0.1,
            "learning_rate": 0.001,
            "use_bn": False
        }
        # 取得測試集 acc/loss 曲線
        test_accs, test_losses = train_with_config(
            default_config,
            num_epochs=num_epochs,
            batch_size=batch_size,
            earlystop=earlystop,
            dataset_name=dataset_name,
            optimizer=optimizer
        )

        # 畫圖
        plt.figure()
        plt.plot(test_accs, label="Test Accuracy")
        plt.plot(test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.title("MNIST Test Accuracy & Loss")
        plt.savefig("mnist_test_acc_loss.png")
        plt.show()

    elif mode == "gga":
        best_config = run_gga(pop_size=pop_size, generations=generations, dataset_name=dataset_name, optimizer=optimizer)
        trainornot = "y"
        if trainornot == "y":
            print("\n🎯 使用最佳參數進行完整訓練")
            test_accs, test_losses = train_with_config(
                best_config,
                num_epochs=num_epochs,
                batch_size=batch_size,
                earlystop=earlystop,
                dataset_name=dataset_name,
                optimizer=optimizer
            )
            plt.figure()
            plt.plot(test_accs, label="Test Accuracy")
            plt.plot(test_losses, label="Test Loss")
            plt.xlabel("Epoch")
            plt.legend()
            plt.title("MNIST Test Accuracy & Loss")
            plt.savefig("mnist_test_acc_loss.png")
            plt.show()
        else:
            print("\n🎯 GGA結束 不進行完整訓練")

if __name__ == "__main__":
    main()