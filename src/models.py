# =============================================================================
# src/models.py — 共享数据模型
# =============================================================================
# 作用：定义全项目通用的数据结构，Phase 1 使用 dataclass 轻量实现，
#       Phase 4 引入 FastAPI 后可无缝切换为 Pydantic BaseModel。
#
# 拓展接口说明：
#   - 所有模型均预留了未来阶段需要的字段（如 clause_id、metadata）
#   - Phase 2 (RAG): Clause 可直接存入 Chroma 作为文档单元
#   - Phase 3 (Agent): AuditResult 作为 Agent 输出标准格式
#   - Phase 4 (FastAPI): 可转换为 Pydantic 请求/响应模型
#   - Phase 5 (Streamlit): 可用于前端数据渲染
# =============================================================================

from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from typing import Optional


# ===================== 枚举定义 =====================

class ContractType(str, enum.Enum):
    """合同类型枚举 — 与 config/rules.yaml 中的 contract_types 键名保持一致"""
    FIXED = "fixed"
    COMMISSION = "commission"
    HYBRID = "hybrid"


class IssueSeverity(str, enum.Enum):
    """异常严重级别"""
    HIGH = "high"        # 金额差异大，必须处理
    MEDIUM = "medium"    # 金额差异中等，需关注
    LOW = "low"          # 轻微差异或提示信息


class AuditStatus(str, enum.Enum):
    """稽核状态"""
    NORMAL = "normal"       # 无异常
    ABNORMAL = "abnormal"   # 存在异常
    ERROR = "error"         # 稽核过程出错


# ===================== 合同相关模型 =====================

@dataclass
class Clause:
    """
    合同条款 —— 最小粒度知识单元

    【Phase 2 拓展位】
    此结构可直接作为 Chroma 向量库的文档单元：
    - clause_id → Chroma 的 id
    - description + clause_type → 文本内容
    - parameters → metadata 中的结构化字段
    - merchant_id / contract_id → metadata 过滤键

    【Phase 3 拓展位】
    clause_type 可作为 Agent 工具调用的参数（extract_clause 的 clause_type 入参）
    """
    clause_id: str
    clause_type: str
    description: str
    contract_id: str
    merchant_id: str
    parameters: dict = field(default_factory=dict)

    def to_document_text(self) -> str:
        """将条款转为文本段落（供 Phase 2 向量化使用）"""
        param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return (
            f"合同 {self.contract_id} | 商户 {self.merchant_id} | "
            f"条款类型: {self.clause_type}\n"
            f"条款内容: {self.description}\n"
            f"参数: {param_str}"
        )


@dataclass
class Contract:
    """
    商户合同 —— 核心业务实体

    【Phase 2 拓展位】
    clauses 列表可逐条写入 Chroma 向量库。
    """
    contract_id: str
    merchant_id: str
    merchant_name: str
    type: ContractType
    fixed_amount: Optional[float] = None
    commission_rate: float = 0.0
    min_guarantee: Optional[float] = None
    billing_cycle: str = "monthly"
    submit_deadline_day: int = 5
    effective_date: str = ""
    end_date: str = ""
    clauses: list[Clause] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转为字典（用于 JSON 序列化 / 数据导出）"""
        return {
            "contract_id": self.contract_id,
            "merchant_id": self.merchant_id,
            "merchant_name": self.merchant_name,
            "type": self.type.value,
            "fixed_amount": self.fixed_amount,
            "commission_rate": self.commission_rate,
            "min_guarantee": self.min_guarantee,
            "billing_cycle": self.billing_cycle,
            "submit_deadline_day": self.submit_deadline_day,
            "effective_date": self.effective_date,
            "end_date": self.end_date,
            "clauses": [
                {
                    "clause_id": c.clause_id,
                    "clause_type": c.clause_type,
                    "description": c.description,
                    "parameters": c.parameters,
                }
                for c in self.clauses
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Contract":
        """从字典构建（用于 JSON 反序列化）"""
        clauses = [
            Clause(
                clause_id=c["clause_id"],
                clause_type=c["clause_type"],
                description=c["description"],
                contract_id=data["contract_id"],
                merchant_id=data["merchant_id"],
                parameters=c.get("parameters", {}),
            )
            for c in data.get("clauses", [])
        ]
        return cls(
            contract_id=data["contract_id"],
            merchant_id=data["merchant_id"],
            merchant_name=data["merchant_name"],
            type=ContractType(data["type"]),
            fixed_amount=data.get("fixed_amount"),
            commission_rate=data.get("commission_rate", 0.0),
            min_guarantee=data.get("min_guarantee"),
            billing_cycle=data.get("billing_cycle", "monthly"),
            submit_deadline_day=data.get("submit_deadline_day", 5),
            effective_date=data.get("effective_date", ""),
            end_date=data.get("end_date", ""),
            clauses=clauses,
        )


# ===================== 账单相关模型 =====================

@dataclass
class Bill:
    """
    商户月度账单

    【Phase 1 说明】
    reported_revenue 为商户申报的营业额。
    paid_amount 为商户实际缴纳的金额。
    规则引擎基于 reported_revenue 和合同条款复算应有金额。

    【Phase 3 拓展位】
    此字段与 tools.py 中 query_revenue 的返回值结构一致。
    """
    merchant_id: str
    month: str                       # 格式: "2024-01"
    reported_revenue: float
    paid_amount: float
    submit_date: str                 # 格式: "2024-02-03"
    actual_revenue: Optional[float] = None  # 仅评测时使用（ground truth）

    def to_dict(self) -> dict:
        return {
            "merchant_id": self.merchant_id,
            "month": self.month,
            "reported_revenue": self.reported_revenue,
            "paid_amount": self.paid_amount,
            "submit_date": self.submit_date,
        }


# ===================== 稽核结果模型 =====================

@dataclass
class AuditIssue:
    """
    稽核发现的异常/问题项

    【Phase 3 拓展位】
    此结构将嵌入 Agent 的 state['issues'] 列表。
    rule_check_node 中生成此对象列表。

    【Phase 5 拓展位】
    Streamlit 根据 severity 渲染不同颜色标签。
    """
    merchant_id: str
    month: str
    issue_type: str          # amount_mismatch / late_submission / revenue_anomaly
    clause_id: str           # 引用的合同条款编号
    expected_value: float
    actual_value: float
    description: str         # 人类可读的描述
    severity: str = "medium"

    def to_dict(self) -> dict:
        return {
            "merchant_id": self.merchant_id,
            "month": self.month,
            "issue_type": self.issue_type,
            "clause_id": self.clause_id,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "description": self.description,
            "severity": self.severity,
        }


@dataclass
class CheckStep:
    """
    稽核过程中的单个检查步骤记录

    【Phase 5 拓展位】
    Streamlit 使用此列表分步展示推理过程（合同类型 → 工具调用 → 规则校验 → 结论）
    """
    step_name: str
    status: str              # success / warning / error / skip
    detail: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step_name": self.step_name,
            "status": self.status,
            "detail": self.detail,
            "data": self.data,
        }


@dataclass
class AuditResult:
    """
    单次稽核完整结果

    这是整个系统的核心输出，所有阶段都以此格式交换数据。

    【Phase 3 拓展位】Agent 的 report_node 构建此对象。
    【Phase 4 拓展位】FastAPI 返回此对象的 JSON 序列化。
    【Phase 5 拓展位】Streamlit 渲染此对象的各字段。
    """
    merchant_id: str
    month: str
    contract_type: str
    status: AuditStatus
    issues: list[AuditIssue] = field(default_factory=list)
    steps: list[CheckStep] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "merchant_id": self.merchant_id,
            "month": self.month,
            "contract_type": self.contract_type,
            "status": self.status.value,
            "issues": [i.to_dict() for i in self.issues],
            "steps": [s.to_dict() for s in self.steps],
            "summary": self.summary,
        }

    def has_issues(self) -> bool:
        return len(self.issues) > 0

    def issue_count(self) -> int:
        return len(self.issues)


# ===================== 工具函数 =====================

def month_to_datetime_str(year: int, month: int) -> str:
    """将年月转为日期字符串，用于 deadline 计算"""
    return f"{year:04d}-{month:02d}-01"


def next_month(year: int, month: int) -> tuple[int, int]:
    """获取下一个月（处理跨年）"""
    if month == 12:
        return year + 1, 1
    return year, month + 1


# ===================== 序列化兼容 =====================

# 提供 to_dict 的混合函数，方便没有 dataclass 的模块使用
def audit_result_to_dict(result: AuditResult) -> dict:
    return result.to_dict()


def dict_to_audit_result(data: dict) -> AuditResult:
    """从字典反序列化为 AuditResult"""
    issues = [
        AuditIssue(
            merchant_id=i["merchant_id"],
            month=i["month"],
            issue_type=i["issue_type"],
            clause_id=i["clause_id"],
            expected_value=i["expected_value"],
            actual_value=i["actual_value"],
            description=i["description"],
            severity=i.get("severity", "medium"),
        )
        for i in data.get("issues", [])
    ]
    steps = [
        CheckStep(
            step_name=s["step_name"],
            status=s["status"],
            detail=s["detail"],
            data=s.get("data", {}),
        )
        for s in data.get("steps", [])
    ]
    return AuditResult(
        merchant_id=data["merchant_id"],
        month=data["month"],
        contract_type=data["contract_type"],
        status=AuditStatus(data.get("status", "normal")),
        issues=issues,
        steps=steps,
        summary=data.get("summary", ""),
    )
