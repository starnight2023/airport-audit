# =============================================================================
# src/__init__.py — 项目包初始化
# =============================================================================
# 使 src 成为 Python 包，方便各模块间相互引用。
#
# 导入方式：
#   from src.models import Contract, Bill, AuditResult
#   from src.rule_engine import audit_single, batch_audit
#   from src.generate_data import load_contracts, load_bills
# =============================================================================

# 版本标识（语义化版本）
__version__ = "1.0.0"
__phase__ = 1
