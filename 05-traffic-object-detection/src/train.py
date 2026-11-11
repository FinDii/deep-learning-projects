"""
Part I: YOLO26 Training Script
"""

import os
from ultralytics import YOLO

def main():

    # 加载模型
    model = YOLO("yolo26n.yaml")

    # 训练
    results = model.train(
        data="./d2-city.yaml",  # 数据集配置文件
        save_dir="./runs",
        pretrained=False,  # 不加载预训练模型
        # 此处可以指定超参数
        amp=False,  # 是否开启 Automatic Mixed Precision (AMP)
        epochs=100,  # 训练轮数 (默认: 100)
    )

    print("Training complete.")
    if results and hasattr(results, 'save_dir'):
        save_dir = results.save_dir
        print(f"\n训练结果保存在: {save_dir}")
        print("生成的文件:")
        for file in os.listdir(save_dir):
            print(f"  - {file}")

    # 在验证集上评估模型
    metrics = model.val()

if __name__ == "__main__":
    main()
