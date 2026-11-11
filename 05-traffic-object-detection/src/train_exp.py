# train_exp.py 

import os
import sys
from ultralytics import YOLO

"""
Part II: YOLO26s - 更换为更大模型并加载 COCO 预训练权重
"""

def experiment_1():
    """实验1: YOLO26s + pretrained (640px)"""
    model = YOLO("yolo26s.pt")          # 加载预训练权重

    model.train(
        data="./d2-city.yaml",
        epochs=150,
        patience=50,
        cache=True,        
        imgsz=640,
        batch=16,          # 在显存允许下batch 越大，GPU 利用率越高。对于 26s，若开启 AMP，batch 可尝试 16。
        amp=True,
        rect=True,         # 矩形训练减少填充，提高训练效率
        project=".",            # 保存根目录
        name="exp1"             # 子目录
    )

    metrics = model.val()
    print(f"Exp1 mAP50-95: {metrics.box.map:.4f}")
    print(f"Exp1 mAP50: {metrics.box.map50:.4f}")

"""
Part III: YOLO26n + 预训练权重 + 更高分辨率且数据增强
"""

def experiment_2():
    """实验2: YOLO26n + pretrained + 高分辨率960px + 数据增强"""
    model = YOLO("yolo26n.pt")

    model.train(
        data="./d2-city.yaml",
        epochs=100,
        cache=False,        # Kaggle 关闭缓存
        imgsz=960,
        batch=10,
        rect=True,          # 矩形训练，减少填充
        mosaic=1.0,
        mixup=0.5,
        copy_paste=0.5,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.4,
        amp=False,
        project=".",
        name="exp2"
    )
    metrics = model.val()
    print(f"Exp2 mAP50-95: {metrics.box.map:.4f}")
    print(f"Exp2 mAP50: {metrics.box.map50:.4f}")

"""
Part IV: YOLO26m + 预训练权重 + 更高分辨率且数据增强、优化器等
"""

def experiment_3():
    """
    实验3: 激进方案 —— 更大模型 + 高分辨率 + 全面优化
    - 模型: YOLO26m (参数量 ~25M)
    - 分辨率: 960px
    - 优化器: AdamW + 余弦退火
    - 训练轮数: 150, 早停 patience=50
    - 增强: mosaic, mixup, copy_paste, flipud, degrees, scale
    - 标签平滑: 0.1
    - 验证时启用 TTA
    """
    model = YOLO("yolo26m.pt")          # 最大的 YOLO26 模型

    model.train(
        data="./d2-city.yaml",
        cache=True,
        epochs=150,
        patience=50,
        imgsz=800,           # 降低至 800，平衡精度与速度
        batch=8,             # 开启AMP
        rect=True,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        label_smoothing=0.1,
        mosaic=0.8,
        mixup=0.4,
        copy_paste=0.5,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.4,
        flipud=0.5,
        degrees=5.0,
        scale=0.5,
        amp=True,               # 若显存不足可开启 amp=True 节省显存
        project=".",
        name="exp3"
    )

    # TTA 验证
    metrics = model.val(augment=True)
    print(f"Exp3 mAP50-95: {metrics.box.map:.4f}")
    print(f"Exp3 mAP50: {metrics.box.map50:.4f}")

"""
Part V: YOLOv11n - 速度与效率对照实验
"""

def experiment_4():
    """实验4: YOLOv11n - 速度与效率对照实验"""
    model = YOLO("yolo11n.pt")  # 或 yolo11s.pt

    model.train(
        data="./d2-city.yaml",
        epochs=150,             # 与实验2/3保持一致
        patience=50,
        cache=True,
        imgsz=800,              # 从标准分辨率开始
        batch=16,
        amp=True,               # 启用混合精度训练
        rect=True,
        project=".",
        name="exp4"
    )

    metrics = model.val()
    print(f"Exp4 (YOLOv11n) mAP50-95: {metrics.box.map:.4f}")
    print(f"Exp4 (YOLOv11n) mAP50: {metrics.box.map50:.4f}")

"""
Part VI: RT-DETR - Transformer 架构实验
"""

def experiment_5():
    """实验5: RT-DETR-L 在 D2-City 数据集上的表现"""
    from ultralytics import RTDETR

    # 使用较大的 RT-DETR-L 模型，以获得更好的性能
    model = RTDETR("rtdetr-l.pt")

    model.train(
        data="./d2-city.yaml",
        epochs=100,
        # patience=20,
        imgsz=512,
        batch=12,            # Transformer 模型通常更耗显存，建议从较小的 batch size 开始
        amp=True,
        cache=True,                 # 缓存图像到内存/显存，加速 IO
        workers=8,                  # 增加数据加载线程
        freeze=10,                  # 冻结骨干网络前 10 层（减少可训练参数）
        save_period=10,             # 每 10 轮保存检查点（满足断点需求）
        project=".",
        name="exp5"
    )

    metrics = model.val()
    print(f"Exp5 (RT-DETR-L) mAP50-95: {metrics.box.map:.4f}")
    print(f"Exp5 (RT-DETR-L) mAP50: {metrics.box.map50:.4f}")

if __name__ == "__main__":
    # 通过命令行参数选择实验
    # 用法: python train_exp.py [1|2|3]  默认运行实验1
    if len(sys.argv) > 1:
        if sys.argv[1] == "2":
            experiment_2()
        elif sys.argv[1] == "3":
            experiment_3()
        elif sys.argv[1] == "4":
            experiment_4()
        elif sys.argv[1] == "5":
            experiment_5()
        else:
            print(f"未知参数: {sys.argv[1]}, 使用默认实验1")
            experiment_1()
    else:
        experiment_1()
