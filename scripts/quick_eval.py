# =============================================================================
# scripts/quick_eval.py — 第一阶段快速评测脚本
# =============================================================================
# 功能说明：
#   1. 自动运行数据生成（或加载已有数据）
#   2. 执行全量稽核
#   3. 加载异常标注 → 计算量化指标
#   4. 打印详细评测报告
#
# 运行方式：
#   python scripts/quick_eval.py                          # 使用现有数据
#   python scripts/quick_eval.py --regenerate             # 重新生成数据再评测
#   python scripts/quick_eval.py --verbose                # 打印每项异常的详情
#
# 【面试问答准备】
# Q: "评测指标有哪些？怎么算的？"
# A: ① 金额复算准确率 = 金额比对正确的账单数 / 总账单数
#    ② 异常识别召回率 = 被标记的异常数 / 总异常数
#    ③ 逾期检测准确率 = 逾期判对账单数 / 需判断逾期账单数
#    ④ 综合异常检测F1 = 2 * (精确率 * 召回率) / (精确率 + 召回率)
# =============================================================================

import os
import sys
import json

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from generate_data import (
    main as generate_data,
    load_contracts,
    load_truth_labels,
    load_bills,
)
from rule_engine import (
    batch_audit,
    audit_single,
    load_rules_config,
    print_audit_summary,
    export_audit_report,
)
from models import AuditStatus

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONTRACTS_PATH = os.path.join(DATA_DIR, "contracts.json")
TRUTH_LABELS_PATH = os.path.join(DATA_DIR, "truth_labels.json")


# ===========================================================================
# 评测指标计算
# ===========================================================================

def evaluate_anomaly_detection(
    audit_results: list,
    truth_labels: list[dict],
    verbose: bool = False,
) -> dict:
    """
    评估异常检测效果（与标注数据对比）

    Args:
        audit_results: 规则引擎输出的稽核结果列表
        truth_labels: 异常标注列表（来自 generate_data 的 ground truth）
        verbose: 是否打印详细对比

    Returns:
        dict: {
            "total_bills": int,
            "true_positives": int,      # 检出且确实是异常的
            "false_positives": int,     # 误报（正常被标记为异常）
            "false_negatives": int,     # 漏报（异常未被检出）
            "true_negatives": int,      # 正确识别为正常的
            "precision": float,         # 精确率
            "recall": float,            # 召回率
            "f1_score": float,          # F1
            "accuracy": float,          # 准确率
        }

    【Phase 5 拓展位】
    此函数将被 evaluate.py 复用并扩展：
    - 增加纯RAG基线对比
    - 增加"条款引用准确率"
    - 增加"无依据回答控制率"
    """
    # 构建标注索引：(merchant_id, month) → anomaly_type_set
    label_index = {}
    for lbl in truth_labels:
        key = (lbl["merchant_id"], lbl["month"])
        if key not in label_index:
            label_index[key] = set()
        label_index[key].add(lbl["anomaly_type"])

    # 统计
    total = len(audit_results)
    tp = fp = fn = tn = 0

    false_negatives_detail = []
    false_positives_detail = []

    for r in audit_results:
        key = (r.merchant_id, r.month)
        is_actual_anomaly = key in label_index
        is_predicted_anomaly = (r.status == AuditStatus.ABNORMAL)

        if is_actual_anomaly and is_predicted_anomaly:
            tp += 1
        elif not is_actual_anomaly and is_predicted_anomaly:
            fp += 1
            false_positives_detail.append({
                "merchant_id": r.merchant_id,
                "month": r.month,
                "issues": [i.description for i in r.issues],
            })
        elif is_actual_anomaly and not is_predicted_anomaly:
            fn += 1
            false_negatives_detail.append({
                "merchant_id": r.merchant_id,
                "month": r.month,
                "expected": label_index.get(key, set()),
            })
        else:
            tn += 1

    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    if verbose:
        if false_negatives_detail:
            print("\n  ⚠ 漏报明细:")
            for fn_item in false_negatives_detail:
                print(f"    商户 {fn_item['merchant_id']} {fn_item['month']}: "
                      f"期望异常 {fn_item['expected']}")
        if false_positives_detail:
            print("\n  ⚠ 误报明细:")
            for fp_item in false_positives_detail:
                print(f"    商户 {fp_item['merchant_id']} {fp_item['month']}: "
                      f"检出异常 {'; '.join(fp_item['issues'])}")

    return {
        "total_bills": total,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def evaluate_amount_accuracy(
    audit_results: list,
    truth_labels: list[dict],
) -> dict:
    """
    评估金额复算准确率

    逻辑：对每条包含金额校验步骤的结果，检查是否有 amount_mismatch。
    如果实际金额一致但系统误报 mismatch，算作"金额复算错误"；
    如果实际金额不一致但系统正确检出 mismatch，算作"金额复算正确"。

    Returns:
        dict: {"correct": int, "incorrect": int, "accuracy": float}
    """
    # 构建标注索引：(merchant_id, month) → 是否金额异常
    amount_anomaly_keys = set()
    for lbl in truth_labels:
        if lbl["anomaly_type"] in ("少报营业额", "金额不符"):
            amount_anomaly_keys.add((lbl["merchant_id"], lbl["month"]))

    correct = 0
    incorrect = 0

    for r in audit_results:
        key = (r.merchant_id, r.month)
        has_amount_issue = any(
            i.issue_type == "amount_mismatch" for i in r.issues
        )
        is_actual_amount_anomaly = key in amount_anomaly_keys

        # 规则引擎正确判断金额 = 有异常→检出 或 无异常→未检出
        if is_actual_amount_anomaly == has_amount_issue:
            correct += 1
        else:
            incorrect += 1

    return {
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": round(correct / (correct + incorrect), 4)
        if (correct + incorrect) > 0 else 0.0,
    }


def evaluate_deadline_accuracy(
    audit_results: list,
    truth_labels: list[dict],
) -> dict:
    """
    评估逾期检测准确率

    Returns:
        dict: {"correct": int, "incorrect": int, "accuracy": float}
    """
    late_keys = set()
    for lbl in truth_labels:
        if lbl["anomaly_type"] == "逾期提交":
            late_keys.add((lbl["merchant_id"], lbl["month"]))

    correct = 0
    incorrect = 0

    for r in audit_results:
        key = (r.merchant_id, r.month)
        has_late_issue = any(
            i.issue_type == "late_submission" for i in r.issues
        )
        is_actual_late = key in late_keys

        if is_actual_late == has_late_issue:
            correct += 1
        else:
            incorrect += 1

    return {
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": round(correct / (correct + incorrect), 4)
        if (correct + incorrect) > 0 else 0.0,
    }


# ===========================================================================
# 评测报告生成
# ===========================================================================

def print_evaluation_report(
    detection: dict,
    amount_acc: dict,
    deadline_acc: dict,
):
    """打印完整的评测报告"""
    print("\n")
    print("=" * 60)
    print("📊 第一阶段评测报告")
    print("=" * 60)

    # 异常检测指标
    print(f"\n【异常检测能力】")
    print(f"  总账单数:         {detection['total_bills']}")
    print(f"  TP (正确检出):    {detection['true_positives']}")
    print(f"  FP (误报):        {detection['false_positives']}")
    print(f"  FN (漏报):        {detection['false_negatives']}")
    print(f"  TN (正确排除):    {detection['true_negatives']}")
    print(f"  ─────────────────────────────")
    print(f"  精确率 (Precision):  {detection['precision']:.2%}")
    print(f"  召回率 (Recall):     {detection['recall']:.2%}")
    print(f"  F1 分数:             {detection['f1_score']:.2%}")
    print(f"  准确率 (Accuracy):   {detection['accuracy']:.2%}")

    # 金额复算
    print(f"\n【金额复算准确率】")
    print(f"  正确:   {amount_acc['correct']}")
    print(f"  错误:   {amount_acc['incorrect']}")
    print(f"  准确率: {amount_acc['accuracy']:.2%}")

    # 逾期检测
    print(f"\n【逾期检测准确率】")
    print(f"  正确:   {deadline_acc['correct']}")
    print(f"  错误:   {deadline_acc['incorrect']}")
    print(f"  准确率: {deadline_acc['accuracy']:.2%}")

    print("\n" + "=" * 60)
    print("✅ 评测完成")
    print("=" * 60)


def save_evaluation_report(
    detection: dict,
    amount_acc: dict,
    deadline_acc: dict,
    output_path: str,
):
    """保存评测报告为 JSON"""
    report = {
        "phase": 1,
        "metrics": {
            "anomaly_detection": detection,
            "amount_accuracy": amount_acc,
            "deadline_accuracy": deadline_acc,
        },
        "summary": {
            "overall_accuracy": detection["accuracy"],
            "f1_score": detection["f1_score"],
            "amount_accuracy": amount_acc["accuracy"],
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"✓ 评测报告已保存: {output_path}")


# ===========================================================================
# 主入口
# ===========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="第一阶段快速评测脚本")
    parser.add_argument("--regenerate", action="store_true", help="重新生成模拟数据")
    parser.add_argument("--verbose", action="store_true", help="打印详细对比信息")
    parser.add_argument("--export", type=str, default=None, help="保存评测报告路径")
    args = parser.parse_args()

    # Step 1: 数据准备
    print("🚀 第一阶段评测启动")
    if args.regenerate or not os.path.exists(CONTRACTS_PATH):
        print("📦 生成模拟数据...")
        generate_data()
    else:
        print("📂 使用已有数据 (使用 --regenerate 可重新生成)")

    # Step 2: 加载数据
    print("\n📥 加载数据...")
    contracts = load_contracts()
    truth_labels = load_truth_labels()
    config = load_rules_config()
    print(f"  合同数: {len(contracts)} | 标注异常数: {len(truth_labels)}")

    # Step 3: 执行全量稽核
    print("\n🔍 执行全量稽核...")
    results = batch_audit(
        contracts=contracts,
        config=config,
    )
    print(f"  稽核完成: {len(results)} 条账单")

    # Step 4: 统计量输出
    print_audit_summary(results)

    # Step 5: 计算评测指标
    print("\n📐 计算评测指标...")
    detection = evaluate_anomaly_detection(
        results, truth_labels, verbose=args.verbose
    )
    amount_acc = evaluate_amount_accuracy(results, truth_labels)
    deadline_acc = evaluate_deadline_accuracy(results, truth_labels)

    # Step 6: 打印与保存报告
    print_evaluation_report(detection, amount_acc, deadline_acc)

    if args.export:
        save_evaluation_report(detection, amount_acc, deadline_acc, args.export)

    # Step 7: 汇总输出
    print(f"\n📈 关键指标摘要:")
    print(f"  F1 Score:        {detection['f1_score']:.2%}")
    print(f"  金额复算准确率:  {amount_acc['accuracy']:.2%}")
    print(f"  逾期检测准确率:  {deadline_acc['accuracy']:.2%}")

    return {
        "detection": detection,
        "amount_accuracy": amount_acc,
        "deadline_accuracy": deadline_acc,
    }


if __name__ == "__main__":
    main()
