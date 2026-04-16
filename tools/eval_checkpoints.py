import os
import torch

from src.core.yaml_config import YAMLConfig
from src.misc import dist_utils
from src.solver import TASKS


def eval_single_checkpoint(config_path, weight_path):
    print(f"\n🔍 Evaluating checkpoint: {weight_path}")

    # ✅ 正确注入 resume 和 test_only 参数，确保加载权重 + 启动推理模式
    cfg = YAMLConfig(config_path, resume=weight_path, test_only=True)

    # 创建求解器
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    # 执行验证
    result = solver.val()

    # 防御性检查
    if result is None or not isinstance(result, tuple) or len(result) < 1:
        print(f"⚠️  No valid COCO evaluation results for {weight_path}")
        return

    stats = result[0]  # 返回的是 (stats, coco_evaluator)

    if stats and "coco_eval_bbox" in stats:
        coco_stats = stats["coco_eval_bbox"]
        ap50 = coco_stats[1]       # AP@IoU=0.50
        ar100 = coco_stats[8]      # AR@maxDets=100

        # 简单 F1 分数估算
        if (ap50 + ar100) > 0:
            f1_score = 2 * (ap50 * ar100) / (ap50 + ar100)
        else:
            f1_score = 0.0

        print(f"\n📊 Evaluation for {os.path.basename(weight_path)}:")
        print(f"   AP50      : {ap50:.4f}")
        print(f"   Recall@100: {ar100:.4f}")
        print(f"   F1 Score  : {f1_score:.4f}")
    else:
        print(f"⚠️  No valid COCO evaluation results for {weight_path}")


if __name__ == "__main__":

    # ✅ 配置路径和要验证的权重文件vis
    config_path = "D:/pythoncache/dual_modal_insulator_fault_detection/GhostNetV2_rtdetrv2_SE/configs/rtdetrv2/rtdetrv2_r50vd_vis_120e_coco.yml"

    weight_paths = [
          "D:/pythoncache/dual_modal_insulator_fault_detection/GhostNetV2_rtdetrv2_SE/tools/output/rtdetrv2_r50vd_vis_10_120e_coco/checkpoint0028.pth"

    ]

    # 初始化分布式（单卡也必须执行）
    dist_utils.setup_distributed(print_rank=0, print_method="builtin")

    # 批量评估
    for weight_path in weight_paths:
        eval_single_checkpoint(config_path, weight_path)

    dist_utils.cleanup()
