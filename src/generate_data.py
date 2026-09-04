# =============================================================================
# src/generate_data.py — 第一阶段：模拟数据生成
# =============================================================================
# 功能说明：
#   1. 定义20家机场商户，随机分配固定租金/营业额提成/保底+提成三种合同
#   2. 为每家商户生成12个月的月度账单 CSV
#   3. 随机选择10%的账单注入三种异常之一：
#      - 少报营业额（reported_revenue 打 8 折）
#      - 金额不符（paid_amount 与实际计算值不一致）
#      - 逾期提交（submit_date 设在截止日后）
#   4. 同时输出一份异常标注文件（truth_labels.json），供评测阶段对比
#
# 运行方式：
#   python src/generate_data.py
#
# 输出文件：
#   data/contracts.json           — 20家商户的合同数据（含条款）
#   data/merchant_info.json       — 商户基础信息
#   data/bills/merchant_{id}_2024.csv — 每家商户的12个月账单
#   data/truth_labels.json        — 异常标注（用于评测对比）
#
# 拓展接口说明：
#   - get_contracts() 导出合约列表 → Phase 2 可由此构建向量知识库
#   - get_bills() 导出账单列表 → Phase 3 可由 query_revenue 工具调用
#   - get_truth_labels() 导出标注 → Phase 5 可作评测基准确
# =============================================================================

import csv
import json
import os
import random
from datetime import datetime, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# 项目路径配置（Phase 6 可改为环境变量注入）
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BILLS_DIR = os.path.join(DATA_DIR, "bills")
CONTRACTS_PATH = os.path.join(DATA_DIR, "contracts.json")
MERCHANT_INFO_PATH = os.path.join(DATA_DIR, "merchant_info.json")
TRUTH_LABELS_PATH = os.path.join(DATA_DIR, "truth_labels.json")

# 确保目录存在
os.makedirs(BILLS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 一、随机种子（保证可复现）
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42


def set_random_seed(seed: int = DEFAULT_SEED):
    """设置随机种子，使数据生成可复现"""
    random.seed(seed)


# ---------------------------------------------------------------------------
# 二、商户与合同定义
# ---------------------------------------------------------------------------

# 真实感商户数据（机场常见商户类型，共 50 家）
MERCHANT_TEMPLATES = [
    # 餐饮 18 家
    {"id": "M001", "name": "云松咖啡", "category": "餐饮"},
    {"id": "M002", "name": "麦香餐厅", "category": "餐饮"},
    {"id": "M003", "name": "星野小馆", "category": "餐饮"},
    {"id": "M004", "name": "青竹咖啡", "category": "餐饮"},
    {"id": "M005", "name": "山泉快餐", "category": "餐饮"},
    {"id": "M006", "name": "长河面馆", "category": "餐饮"},
    {"id": "M007", "name": "满庭茶点", "category": "餐饮"},
    {"id": "M008", "name": "湖畔米粉", "category": "餐饮"},
    {"id": "M018", "name": "暖阳小吃", "category": "餐饮"},
    {"id": "M021", "name": "林间火锅", "category": "餐饮"},
    {"id": "M022", "name": "谷雨夹馍", "category": "餐饮"},
    {"id": "M023", "name": "禾田鸡煲", "category": "餐饮"},
    {"id": "M024", "name": "溪岸捞面", "category": "餐饮"},
    {"id": "M025", "name": "初见小面", "category": "餐饮"},
    {"id": "M026", "name": "金穗牛肉面", "category": "餐饮"},
    {"id": "M027", "name": "炭香烤肉", "category": "餐饮"},
    {"id": "M028", "name": "渔火酸菜鱼", "category": "餐饮"},
    {"id": "M029", "name": "芳草米线", "category": "餐饮"},
    # 零售 18 家
    {"id": "M009", "name": "香韵卤味", "category": "零售"},
    {"id": "M010", "name": "川香鸭脖", "category": "零售"},
    {"id": "M011", "name": "山果铺子", "category": "零售"},
    {"id": "M012", "name": "拾光杂货", "category": "零售"},
    {"id": "M013", "name": "怡美日用", "category": "零售"},
    {"id": "M019", "name": "徽味卤坊", "category": "零售"},
    {"id": "M020", "name": "果乐零食", "category": "零售"},
    {"id": "M030", "name": "松栗干果", "category": "零售"},
    {"id": "M031", "name": "青禾零食", "category": "零售"},
    {"id": "M032", "name": "萌趣潮玩", "category": "零售"},
    {"id": "M033", "name": "智享数码", "category": "零售"},
    {"id": "M034", "name": "峰行数码", "category": "零售"},
    {"id": "M035", "name": "简尚服饰", "category": "零售"},
    {"id": "M036", "name": "织语男装", "category": "零售"},
    {"id": "M037", "name": "万家生鲜", "category": "零售"},
    {"id": "M038", "name": "晨光便利店", "category": "零售"},
    {"id": "M039", "name": "邻里便利店", "category": "零售"},
    {"id": "M040", "name": "迅捷便利", "category": "零售"},
    # 饮品 9 家
    {"id": "M014", "name": "仙草茶饮", "category": "饮品"},
    {"id": "M015", "name": "甜橙茶饮", "category": "饮品"},
    {"id": "M016", "name": "芳茶鲜奶", "category": "饮品"},
    {"id": "M017", "name": "蜜语冰饮", "category": "饮品"},
    {"id": "M041", "name": "茗香奶茶", "category": "饮品"},
    {"id": "M042", "name": "城西茶饮", "category": "饮品"},
    {"id": "M043", "name": "乐享茶饮", "category": "饮品"},
    {"id": "M044", "name": "一味茶饮", "category": "饮品"},
    {"id": "M045", "name": "楚风茶香", "category": "饮品"},
    # 服务 5 家
    {"id": "M046", "name": "畅联通讯", "category": "服务"},
    {"id": "M047", "name": "远方快递", "category": "服务"},
    {"id": "M048", "name": "书香书屋", "category": "服务"},
    {"id": "M049", "name": "轻装寄存", "category": "服务"},
    {"id": "M050", "name": "舒心按摩", "category": "服务"},
]

# 合同类型概率权重（总数50，分配：20 fixed / 15 commission / 15 hybrid）
CONTRACT_TYPE_DISTRIBUTION = ["fixed"] * 20 + ["commission"] * 15 + ["hybrid"] * 15


def assign_contract_types(seed: int = DEFAULT_SEED) -> list[dict]:
    """
    为商户分配合同类型，每条记录包含完整的合同参数

    Returns:
        list of dict: 每个元素是一个商户的合同配置
    """
    local_rng = random.Random(seed)
    # 打乱合同类型分配顺序
    types = CONTRACT_TYPE_DISTRIBUTION.copy()
    local_rng.shuffle(types)

    merchants = []
    for i, tmpl in enumerate(MERCHANT_TEMPLATES):
        ctype = types[i]
        contract = {
            "merchant_id": tmpl["id"],
            "merchant_name": tmpl["name"],
            "category": tmpl["category"],
            "type": ctype,
        }
        # 根据合同类型设定不同参数
        if ctype == "fixed":
            # 固定租金：月租金 5000~20000 不等（依店铺大小）
            contract["fixed_amount"] = round(local_rng.uniform(5000, 20000), 2)
            contract["commission_rate"] = 0.0
            contract["min_guarantee"] = None
        elif ctype == "commission":
            # 营业额提成：提成比例 8%~20%
            contract["fixed_amount"] = None
            contract["commission_rate"] = round(local_rng.uniform(0.08, 0.20), 4)
            contract["min_guarantee"] = None
        else:  # hybrid
            # 保底+提成：保底额 5000~15000，提成比例 10%~18%
            contract["fixed_amount"] = None
            contract["commission_rate"] = round(local_rng.uniform(0.10, 0.18), 4)
            contract["min_guarantee"] = round(local_rng.uniform(5000, 15000), 2)
        merchants.append(contract)
    return merchants


# ---------------------------------------------------------------------------
# 三、账单生成
# ---------------------------------------------------------------------------

# 各商户类型的基础月营业额范围（元）
REVENUE_RANGES = {
    "餐饮": (30000, 120000),
    "零售": (20000, 80000),
    "饮品": (15000, 60000),
    "服务": (10000, 50000),
}


def generate_monthly_bills(
    merchants: list[dict],
    year: int = 2025,
    anomaly_rate: float = 0.10,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict], list[dict]]:
    """
    为所有商户生成12个月账单

    Args:
        merchants: 商户配置列表（含合同参数）
        year: 账单年份
        anomaly_rate: 异常比例（0~1）
        seed: 随机种子

    Returns:
        (bills, truth_labels)
        bills: 所有账单的列表 [{"merchant_id", "month", "reported_revenue", "paid_amount", "submit_date", ...}, ...]
        truth_labels: 异常标注 [{"merchant_id", "month", "anomaly_type", ...}, ...]

    【Phase 3 拓展位】
    truth_labels 可被 evaluate.py 用于评测 Agent 的异常识别能力。
    """
    local_rng = random.Random(seed)
    bills = []
    truth_labels = []

    for merchant in merchants:
        mid = merchant["merchant_id"]
        ctype = merchant["type"]
        category = merchant["category"]
        rev_range = REVENUE_RANGES.get(category, (20000, 80000))

        for month in range(1, 13):
            month_str = f"{year}-{month:02d}"

            # ---- 1. 生成基础营业额（月度有自然波动） ----
            base_revenue = local_rng.uniform(rev_range[0], rev_range[1])
            # 添加季节因子（Q1偏低，Q4偏高）
            season_factor = 1.0 + 0.1 * (month / 12)
            actual_revenue = round(base_revenue * season_factor, 2)

            # ---- 2. 计算应付金额 ----
            if ctype == "fixed":
                expected_paid = merchant["fixed_amount"]
            elif ctype == "commission":
                expected_paid = round(actual_revenue * merchant["commission_rate"], 2)
            else:  # hybrid
                commission_part = round(actual_revenue * merchant["commission_rate"], 2)
                expected_paid = max(merchant["min_guarantee"], commission_part)

            # ---- 3. 默认：正常账单 ----
            reported_revenue = actual_revenue
            paid_amount = expected_paid
            # 提交日期：下月3日前（正常）
            next_m = month + 1 if month < 12 else 1
            next_y = year if month < 12 else year + 1
            submit_date = f"{next_y:04d}-{next_m:02d}-{local_rng.randint(1, 3):02d}"
            is_anomaly = False
            anomaly_type = ""

            # ---- 4. 随机注入异常（10%） ----
            if local_rng.random() < anomaly_rate:
                is_anomaly = True
                anomaly_choice = local_rng.choice([
                    "underreport_revenue",
                    "amount_mismatch",
                    "late_submission",
                ])

                if anomaly_choice == "underreport_revenue":
                    # 少报营业额：reported_revenue 打 8 折
                    # paid_amount 仍基于实际营业额计算（即高于应基于少报额计算的金额）
                    reported_revenue = round(actual_revenue * 0.8, 2)
                    anomaly_type = "少报营业额"

                elif anomaly_choice == "amount_mismatch":
                    # 金额不符：paid_amount 改成随机错误值
                    error_factor = local_rng.choice([0.5, 0.7, 0.85, 1.15, 1.3, 1.5])
                    paid_amount = round(expected_paid * error_factor, 2)
                    anomaly_type = "金额不符"

                else:  # late_submission
                    # 逾期提交：提交日期设为下月10~20号
                    submit_date = f"{next_y:04d}-{next_m:02d}-{local_rng.randint(10, 20):02d}"
                    anomaly_type = "逾期提交"

            bill = {
                "merchant_id": mid,
                "month": month_str,
                "reported_revenue": reported_revenue,
                "paid_amount": paid_amount,
                "submit_date": submit_date,
                # 以下字段为评测用，不写入最终 CSV，但会写入 truth_labels
                "_actual_revenue": actual_revenue,
                "_expected_paid": expected_paid,
                "_is_anomaly": is_anomaly,
                "_anomaly_type": anomaly_type,
            }
            bills.append(bill)

            # 记录标注
            if is_anomaly:
                truth_labels.append({
                    "merchant_id": mid,
                    "merchant_name": merchant["merchant_name"],
                    "month": month_str,
                    "anomaly_type": anomaly_type,
                    "contract_type": ctype,
                    "reported_revenue": reported_revenue,
                    "paid_amount": paid_amount,
                    "expected_paid": expected_paid,
                    "actual_revenue": actual_revenue,
                    "submit_date": submit_date,
                })

    return bills, truth_labels


# ---------------------------------------------------------------------------
# 四、合同条款构建
# ---------------------------------------------------------------------------

def build_contracts_with_clauses(
    merchants: list[dict],
    year: int = 2025,
) -> list[dict]:
    """
    为商户构建包含结构化条款的完整合同数据

    Returns:
        list of dict: 符合 models.Contract 结构的合同数据

    【Phase 2 拓展位】
    返回的 clauses 列表可以直接写入 Chroma 向量库：
    - 每条 clause 是一个独立文档
    - clause_id 作为文档 ID
    - description 作为文档文本
    - parameters 作为 metadata
    """
    contracts = []
    for i, merchant in enumerate(merchants):
        contract_id = f"CTR-{i+1:03d}"
        mid = merchant["merchant_id"]
        ctype = merchant["type"]

        clauses = []

        # ----- 条款1：租金计算方式 -----
        if ctype == "fixed":
            formula_desc = f"固定月租金 {merchant['fixed_amount']:.2f} 元"
            calc_params = {
                "fixed_amount": merchant["fixed_amount"],
            }
        elif ctype == "commission":
            rate_pct = merchant["commission_rate"] * 100
            formula_desc = f"月租金 = 申报营业额 × {rate_pct:.2f}%"
            calc_params = {
                "commission_rate": merchant["commission_rate"],
            }
        else:  # hybrid
            rate_pct = merchant["commission_rate"] * 100
            formula_desc = (
                f"月租金 = max(保底额 {merchant['min_guarantee']:.2f} 元, "
                f"申报营业额 × {rate_pct:.2f}%)"
            )
            calc_params = {
                "commission_rate": merchant["commission_rate"],
                "min_guarantee": merchant["min_guarantee"],
            }

        clauses.append({
            "clause_id": f"{contract_id}-clause-001",
            "clause_type": "rent_calculation",
            "description": f"租金计算方式：{formula_desc}",
            "parameters": calc_params,
        })

        # ----- 条款2：报表提交截止日 -----
        clauses.append({
            "clause_id": f"{contract_id}-clause-002",
            "clause_type": "submission_deadline",
            "description": "商户需在每月5日前提交上月营业报表",
            "parameters": {"deadline_day": 5},
        })

        # ----- 条款3：滞纳金（可选） -----
        clauses.append({
            "clause_id": f"{contract_id}-clause-003",
            "clause_type": "late_fee",
            "description": "逾期提交按日收取滞纳金，日费率0.5%，宽限期3天",
            "parameters": {
                "late_fee_rate": 0.005,
                "grace_days": 3,
            },
        })

        # ----- 条款4：营业额达标线（仅 hybrid 类型） -----
        if ctype == "hybrid":
            threshold_revenue = round(
                merchant["min_guarantee"] / merchant["commission_rate"], 2
            )
            clauses.append({
                "clause_id": f"{contract_id}-clause-004",
                "clause_type": "revenue_threshold",
                "description": (
                    f"营业额达标线为 {threshold_revenue:.2f} 元，"
                    f"超过该值后按提成计算，否则按保底额计算"
                ),
                "parameters": {
                    "threshold_revenue": threshold_revenue,
                    "min_guarantee": merchant["min_guarantee"],
                },
            })

        contract = {
            "contract_id": contract_id,
            "merchant_id": mid,
            "merchant_name": merchant["merchant_name"],
            "type": ctype,
            "category": merchant.get("category", ""),
            "fixed_amount": merchant.get("fixed_amount"),
            "commission_rate": merchant.get("commission_rate", 0.0),
            "min_guarantee": merchant.get("min_guarantee"),
            "billing_cycle": "monthly",
            "submit_deadline_day": 5,
            "effective_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "clauses": clauses,
        }
        contracts.append(contract)

    return contracts


# ---------------------------------------------------------------------------
# 五、导出函数
# ---------------------------------------------------------------------------

def save_contracts(contracts: list[dict], path: str = CONTRACTS_PATH):
    """保存合同 JSON"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(contracts, f, ensure_ascii=False, indent=2)
    print(f"✓ 合同数据已保存: {path} ({len(contracts)} 个合同)")


def save_merchant_info(merchants: list[dict], path: str = MERCHANT_INFO_PATH):
    """保存商户基础信息"""
    info = [
        {
            "merchant_id": m["merchant_id"],
            "merchant_name": m["merchant_name"],
            "category": m.get("category", ""),
            "contract_type": m["type"],
        }
        for m in merchants
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"✓ 商户信息已保存: {path} ({len(info)} 家商户)")


def save_bills_csv(bills: list[dict], bills_dir: str = BILLS_DIR):
    """
    按商户分 CSV 文件保存账单

    【Phase 4 拓展位】
    FastAPI 的 /audit 接口将读取这些 CSV 作为数据源。
    Phase 3 的 query_revenue 工具也读取同一路径。
    """
    # 按商户分组
    merchant_bills: dict[str, list[dict]] = {}
    for bill in bills:
        mid = bill["merchant_id"]
        if mid not in merchant_bills:
            merchant_bills[mid] = []
        merchant_bills[mid].append(bill)

    csv_count = 0
    for mid, mbills in sorted(merchant_bills.items()):
        fpath = os.path.join(bills_dir, f"{mid}_2024.csv")
        with open(fpath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "merchant_id", "month", "reported_revenue",
                "paid_amount", "submit_date",
            ])
            writer.writeheader()
            for bill in sorted(mbills, key=lambda x: x["month"]):
                # 写入 CSV 时去除内部字段（下划线前缀表示内部使用）
                writer.writerow({
                    "merchant_id": bill["merchant_id"],
                    "month": bill["month"],
                    "reported_revenue": bill["reported_revenue"],
                    "paid_amount": bill["paid_amount"],
                    "submit_date": bill["submit_date"],
                })
        csv_count += 1
    print(f"✓ 账单文件已保存: {bills_dir}/ 目录下 ({csv_count} 个文件)")


def save_truth_labels(labels: list[dict], path: str = TRUTH_LABELS_PATH):
    """保存异常标注（用于评测对比）"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"✓ 异常标注已保存: {path} ({len(labels)} 条异常记录)")


# ---------------------------------------------------------------------------
# 六、数据加载函数（供 rule_engine.py 和其他模块使用）
# ---------------------------------------------------------------------------

def load_contracts(path: str = CONTRACTS_PATH) -> list[dict]:
    """加载合同数据"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_merchant_info(path: str = MERCHANT_INFO_PATH) -> list[dict]:
    """加载商户信息"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_truth_labels(path: str = TRUTH_LABELS_PATH) -> list[dict]:
    """加载异常标注"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_bills(
    merchant_id: Optional[str] = None,
    month: Optional[str] = None,
    bills_dir: str = BILLS_DIR,
) -> list[dict]:
    """
    加载账单数据（支持按商户和月份过滤）

    【Phase 3 拓展位】
    此函数作为 query_revenue(merchant_id, month) 工具的后端实现。
    Phase 3 的 Agent 通过此函数获取营收数据。

    Args:
        merchant_id: 可选，仅返回指定商户的账单
        month: 可选，仅返回指定月份的账单
        bills_dir: 账单文件目录

    Returns:
        list of dict: 账单记录列表
    """
    all_bills = []

    if not os.path.exists(bills_dir):
        raise FileNotFoundError(f"账单目录不存在: {bills_dir}")

    for fname in os.listdir(bills_dir):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(bills_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["reported_revenue"] = float(row["reported_revenue"])
                row["paid_amount"] = float(row["paid_amount"])
                all_bills.append(row)

    # 过滤
    if merchant_id:
        all_bills = [b for b in all_bills if b["merchant_id"] == merchant_id]
    if month:
        all_bills = [b for b in all_bills if b["month"] == month]

    return all_bills


# ---------------------------------------------------------------------------
# 七、统计信息
# ---------------------------------------------------------------------------

def print_generation_summary(bills: list[dict], truth_labels: list[dict]):
    """打印数据生成摘要"""
    total = len(bills)
    anomaly_count = len(truth_labels)
    normal_count = total - anomaly_count

    # 异常类型分布
    type_dist = {}
    for label in truth_labels:
        t = label["anomaly_type"]
        type_dist[t] = type_dist.get(t, 0) + 1

    print("\n" + "=" * 50)
    print("📊 数据生成摘要")
    print("=" * 50)
    print(f"  总账单数:   {total}")
    print(f"  正常账单:   {normal_count} ({normal_count/total*100:.1f}%)")
    print(f"  异常账单:   {anomaly_count} ({anomaly_count/total*100:.1f}%)")
    print(f"  异常分布:  {type_dist}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# 八、主入口
# ---------------------------------------------------------------------------

def main(
    seed: int = DEFAULT_SEED,
    year: int = 2025,
    anomaly_rate: float = 0.10,
):
    """执行完整的数据生成流程"""
    print("🚀 开始生成模拟数据...")
    set_random_seed(seed)

    # Step 1: 分配合同
    merchants = assign_contract_types(seed)
    print(f"✓ 已为 {len(merchants)} 家商户分配合同类型")

    # Step 2: 构建合同条款
    contracts = build_contracts_with_clauses(merchants, year=year)
    print(f"✓ 已构建 {len(contracts)} 份合同 (含条款)")

    # Step 3: 生成月度账单
    # 使用偏离种子避免与 assign_contract_types 的随机序列碰撞
    bills_seed = seed + 1000
    bills, truth_labels = generate_monthly_bills(
        merchants, year=year, anomaly_rate=anomaly_rate, seed=bills_seed
    )
    print(f"✓ 已生成 {len(bills)} 条月度账单 ({year}年度)")

    # Step 4: 导出文件
    save_contracts(contracts)
    save_merchant_info(merchants)
    save_bills_csv(bills)
    save_truth_labels(truth_labels)

    # Step 5: 打印摘要
    print_generation_summary(bills, truth_labels)

    return contracts, bills, truth_labels


if __name__ == "__main__":
    main()
