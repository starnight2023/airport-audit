# =============================================================================
# src/app.py — 机场非航收入智能稽核系统 Streamlit 前端
# =============================================================================
# 企业级稽核系统风格：无emoji、颜色标签系统、历史记录、结构化展示
#
# 运行方式：
#   streamlit run src/app.py
# =============================================================================

import json
import os
import sys
import time
from datetime import datetime, date
from typing import Optional

# ---------------------------------------------------------------------------
# .env 环境变量
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Streamlit
# ---------------------------------------------------------------------------
import streamlit as st

# ---------------------------------------------------------------------------
# 项目路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# ===========================================================================
# 一、session_state 初始化
# ===========================================================================

def init_session_state():
    """初始化 session_state"""
    if "audit_history" not in st.session_state:
        st.session_state.audit_history = []


# ===========================================================================
# 二、数据加载
# ===========================================================================

@st.cache_data
def load_merchants() -> list[dict]:
    """加载商户列表"""
    path = os.path.join(DATA_DIR, "merchant_info.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=0)
def load_months() -> list[str]:
    """从账单文件提取可用月份"""
    import csv
    bills_dir = os.path.join(DATA_DIR, "bills")
    months = set()
    for fname in os.listdir(bills_dir):
        if not fname.endswith(".csv"):
            continue
        with open(os.path.join(bills_dir, fname), "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                months.add(row["month"])
    return sorted(list(months))


@st.cache_data
def load_contract_type_map() -> dict:
    """加载商户ID到合同类型的映射"""
    path = os.path.join(DATA_DIR, "contracts.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        contracts = json.load(f)
    return {c["merchant_id"]: c.get("type", "") for c in contracts}


# ===========================================================================
# 三、稽核调用函数
# ===========================================================================

# 不缓存 audit_mock 结果，避免 Streamlit 跨商户/月份返回缓存数据
def _cached_mock_audit(_merchant_id: str, _month: str) -> dict:
    from src.agent_graph import audit_mock
    return audit_mock(_merchant_id, _month)


def run_audit_standard(merchant_id: str, month: str) -> Optional[dict]:
    """标准稽核模式：规则引擎 + 本地检索"""
    try:
        return _cached_mock_audit(merchant_id, month)
    except Exception as e:
        st.error(f"稽核执行失败: {e}")
        return None


def run_audit_deep(merchant_id: str, month: str) -> Optional[dict]:
    """深度稽核模式：含AI分析报告"""
    try:
        from src.agent_graph import audit
        return audit(merchant_id, month)
    except ValueError as e:
        st.error("深度分析功能暂不可用，请切换至标准稽核模式")
        return None
    except Exception as e:
        st.error(f"稽核执行失败: {e}")
        return None


# ===========================================================================
# 四、自然语言解析
# ===========================================================================

def parse_natural_language(nl_text: str) -> Optional[dict]:
    """调用LLM解析自然语言为稽核参数"""
    try:
        from src.agent_graph import parse_nl_query
        return parse_nl_query(nl_text, reference_year=2025)
    except Exception as e:
        st.error(f"语义解析服务暂不可用: {e}")
        return None


# ===========================================================================
# 五、样式定义
# ===========================================================================

def inject_css():
    """注入全局CSS样式"""
    st.markdown("""
    <style>
        /* 全局基础字号放大一号 */
        html { font-size: 20px; }
        /* 页面顶部大标题保持原大 */
        h1 { font-size: 2.0rem; }
        .main > div { padding: 1rem 2rem; }
        .stExpander { border: 1px solid #e0e0e0; border-radius: 6px; margin-bottom: 0.75rem; }
        .stExpander .stMarkdown p, .stExpander .stMarkdown li { font-size: 1rem; }
        .sidebar .stMarkdown p, .sidebar .stMarkdown li { font-size: 0.95rem; }
        .stMetric label { font-size: 0.85rem; font-weight: 500; }
        .stMetric div[data-testid="metric-value"] { font-size: 1.0rem; font-weight: 600; }
        .stExpander summary { font-size: 1.1rem; font-weight: 600; }
        .main table { font-size: 0.9rem; }
        .st-caption p { font-size: 0.85rem !important; }
        /* 状态标签 */
        .tag-normal { color: #2e7d32; font-weight: 600; }
        .tag-abnormal { color: #d32f2f; font-weight: 600; }
        .tag-warning { color: #e65100; font-weight: 600; }
        .tag-info { color: #1565c0; }
        /* 顶部横幅 */
        .banner-pass { background-color: #e8f5e9; color: #1b5e20; padding: 0.75rem 1rem; border-radius: 6px; font-weight: 600; margin-bottom: 1rem; }
        .banner-fail { background-color: #ffebee; color: #b71c1c; padding: 0.75rem 1rem; border-radius: 6px; font-weight: 600; margin-bottom: 1rem; }
        .banner-skip { background-color: #f5f5f5; color: #616161; padding: 0.75rem 1rem; border-radius: 6px; font-weight: 600; margin-bottom: 1rem; }
        /* 异常容器：Streamlit container(border=True) 无边框，通过覆盖添加左边框 */
        div[data-testid="stVerticalBlock"] > div[data-testid="element-container"]:has(.issue-border-high) > div,
        div[data-testid="stVerticalBlock"] > div[data-testid="element-container"]:has(.issue-border-mid) > div,
        div[data-testid="stVerticalBlock"] > div[data-testid="element-container"]:has(.issue-border-low) > div {
            border-left: 4px solid #d32f2f !important;
            padding-left: 0.75rem !important;
            margin-bottom: 0.5rem !important;
        }
        /* 金额差异 */
        .diff-negative { color: #d32f2f; font-weight: 600; }
        .diff-positive { color: #2e7d32; font-weight: 600; }
        .diff-zero { color: #616161; }
        /* 代码块 */
        code.audit-code { font-size: 0.85rem; background: #f5f5f5; padding: 0.2rem 0.4rem; border-radius: 3px; }
        pre.audit-pre { background: #f5f5f5; padding: 0.75rem; border-radius: 4px; font-size: 0.95rem; line-height: 1.6; }
        /* 表头 */
        th { font-weight: 600 !important; font-size: 0.85rem !important; }
        /* 历史记录表格异常行 */
        .row-abnormal td { background-color: #fff5f5 !important; }
    </style>
    """, unsafe_allow_html=True)


# ===========================================================================
# 六、侧边栏
# ===========================================================================

def render_sidebar() -> tuple:
    """
    渲染侧边栏

    Returns:
        (merchant_id, month, run_clicked, is_deep_mode, nl_input, is_nl_active)
    """
    merchants = load_merchants()
    months = load_months()
    ctype_map = load_contract_type_map()
    type_label = {"fixed": "固定租金", "commission": "提成", "hybrid": "保底+提成"}

    with st.sidebar:
        st.header("稽核条件")

        # 输入方式切换
        input_mode = st.radio(
            "输入方式",
            options=["下拉选择", "自然语言查询"],
            index=0,
        )
        is_nl_active = "自然语言" in input_mode

        merchant_id = ""
        month = ""
        nl_input = ""

        if is_nl_active:
            nl_input = st.text_input(
                "查询描述",
                placeholder="例如：查看云松咖啡今年3月的租金缴纳情况",
                label_visibility="collapsed",
            )
            st.caption("描述商户和时间后点击「开始稽核」，系统自动解析执行")
        else:
            # 商户选择
            merchant_options = {}
            for m in merchants:
                mid = m["merchant_id"]
                ct = ctype_map.get(mid, m.get("contract_type", ""))
                ct_zh = type_label.get(ct, ct)
                label = f"{mid} - {m['merchant_name']} ({ct_zh})"
                merchant_options[label] = mid
            selected_label = st.selectbox(
                "选择商户",
                options=list(merchant_options.keys()),
                index=0,
            )
            merchant_id = merchant_options[selected_label]

            # 月份选择
            month = st.selectbox("选择月份", options=months, index=0)

        # 稽核模式（自然语言模式固定深度稽核，不显示选择）
        if is_nl_active:
            is_deep_mode = True
        else:
            audit_mode = st.radio(
                "稽核模式",
                options=["标准稽核（纯 Python 规则引擎）", "深度稽核（调用 DeepSeek LLM）"],
                index=0,
                help="标准稽核：纯 Python 规则引擎复算，不调用 LLM，无需 API Key；深度稽核：调用 DeepSeek LLM 生成分析报告，需配置 DEEPSEEK_API_KEY",
            )
            is_deep_mode = "深度稽核" in audit_mode
            st.caption("标准稽核约 10-15 秒；深度稽核约 20-40 秒（需配置 DEEPSEEK_API_KEY）")

        # 稽核按钮
        run_clicked = st.button("开始稽核", type="primary", use_container_width=True)

    return merchant_id, month, run_clicked, is_deep_mode, nl_input, is_nl_active


# ===========================================================================
# 七、稽核结果展示
# ===========================================================================

def render_history_table():
    """渲染稽核历史记录表格"""
    st.subheader("稽核工作台")

    history = st.session_state.get("audit_history", [])

    if not history:
        st.info("暂无稽核记录，请在左侧选择商户和月份后点击「开始稽核」")
        return

    # 构建表格数据
    rows = []
    for h in reversed(history):  # 最新在前
        status_tag = h.get("status", "normal")
        status_text = {"normal": "正常", "abnormal": "异常", "error": "待核查"}.get(status_tag, "")
        rows.append({
            "商户名称": h.get("merchant_name", ""),
            "月份": h.get("month", ""),
            "合同类型": {"fixed": "固定租金", "commission": "提成", "hybrid": "保底+提成"}.get(
                h.get("contract_type", ""), ""),
            "稽核结论": status_text,
            "异常数": str(h.get("issue_count", 0)),
            "稽核时间": h.get("timestamp", ""),
            "_status": status_tag,
        })

    # 渲染表格（用markdown模拟带颜色的表格）
    table_html = '<table style="width:100%;border-collapse:collapse;font-size:0.9rem;">'
    table_html += '<thead><tr style="border-bottom:2px solid #e0e0e0;">'
    for col in ["商户名称", "月份", "合同类型", "稽核结论", "异常数", "稽核时间"]:
        table_html += f'<th style="padding:0.5rem;text-align:left;color:#616161;">{col}</th>'
    table_html += '</tr></thead><tbody>'

    for row in rows:
        row_class = 'style="background-color:#fff5f5;"' if row["_status"] == "abnormal" else ""
        table_html += f'<tr {row_class} style="border-bottom:1px solid #f0f0f0;">'
        color = {"normal": "#2e7d32", "abnormal": "#d32f2f", "error": "#e65100"}.get(row["_status"], "#000")
        for col in ["商户名称", "月份", "合同类型", "稽核结论", "异常数", "稽核时间"]:
            if col == "稽核结论":
                table_html += f'<td style="padding:0.5rem;color:{color};font-weight:600;">{row[col]}</td>'
            else:
                table_html += f'<td style="padding:0.5rem;">{row[col]}</td>'
        table_html += '</tr>'
    table_html += '</tbody></table>'

    st.markdown(table_html, unsafe_allow_html=True)


def append_history(result: dict, elapsed: float):
    """追加稽核记录到 session_state"""
    contract = result.get("contract", {})
    audit_result = result.get("audit_result", {})
    issues = result.get("issues", [])
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    record = {
        "merchant_id": result.get("merchant_id", ""),
        "merchant_name": contract.get("merchant_name", ""),
        "month": result.get("month", ""),
        "contract_type": contract.get("type", ""),
        "status": audit_result.get("status", "error"),
        "issue_count": len(issues),
        "timestamp": now,
    }

    history = st.session_state.audit_history
    history.append(record)
    # 保留最近20条
    if len(history) > 20:
        history = history[-20:]
    st.session_state.audit_history = history


def render_audit_result(result: dict, elapsed: float, mode: str):
    """渲染稽核结果"""
    if result is None:
        st.error("稽核服务暂不可用，请稍后重试或联系系统管理员")
        return

    issues = result.get("issues", [])
    audit_result = result.get("audit_result", {})
    contract = result.get("contract", {})
    revenue_data = result.get("revenue_data", {})
    report = result.get("report", {})

    merchant_name = contract.get("merchant_name", result.get("merchant_id", ""))
    contract_type = contract.get("type", result.get("contract_type", ""))
    month = result.get("month", "")
    status = audit_result.get("status", "unknown")
    ctype_zh = {"fixed": "固定租金", "commission": "营业额提成", "hybrid": "保底+提成"}

    # ========== 顶部结论横幅 ==========
    if status == "normal":
        st.markdown(
            '<div class="banner-pass">稽核通过：未发现异常</div>',
            unsafe_allow_html=True,
        )
    elif status == "abnormal":
        st.markdown(
            f'<div class="banner-fail">稽核异常：发现 {len(issues)} 项问题，详见下方明细</div>',
            unsafe_allow_html=True,
        )
    else:
        # error 或 unknown
        summary = audit_result.get("summary", "数据不可用，建议人工核查")
        st.markdown(
            f'<div class="banner-skip">{summary}</div>',
            unsafe_allow_html=True,
        )

    # ========== 关键指标行 ==========
    expected_amount = 0.0
    paid_amount = 0.0
    diff_amount = 0.0
    has_amount_data = False

    if revenue_data:
        paid_amount = revenue_data.get("paid_amount", 0.0)
        # 从 audit_result 的 steps 中提取期望金额
        steps = audit_result.get("steps", [])
        for step in steps:
            if step.get("step_name") == "金额校验":
                expected_amount = step.get("data", {}).get("expected_amount", 0.0)
                has_amount_data = True
                break
        if not has_amount_data:
            # 尝试从 issues 中提取
            for issue in issues:
                if issue.get("issue_type") == "amount_mismatch":
                    expected_amount = issue.get("expected_value", 0.0)
                    has_amount_data = True
                    break

    diff_amount = round(paid_amount - expected_amount, 2)

    kcols = st.columns(5)
    with kcols[0]:
        st.metric("商户", merchant_name if merchant_name else result.get("merchant_id", ""))
    with kcols[1]:
        st.metric("合同类型", ctype_zh.get(contract_type, contract_type))
    with kcols[2]:
        val = f"¥{expected_amount:,.2f}" if expected_amount else "—"
        st.metric("应缴金额", val)
    with kcols[3]:
        val = f"¥{paid_amount:,.2f}" if paid_amount else "—"
        st.metric("实缴金额", val)
    with kcols[4]:
        if has_amount_data:
            diff_color = "diff-negative" if diff_amount < 0 else ("diff-positive" if diff_amount > 0 else "diff-zero")
            st.metric("差额", f"¥{diff_amount:+,.2f}")
        else:
            st.metric("差额", "—")

    st.divider()

    
    # =========================================================================
    # 二、稽核依据（合并合同信息 + 营收数据）
    # =========================================================================
    with st.expander("稽核依据", expanded=False):
        st.markdown("**商户基本信息**")
        cid = contract.get("contract_id", "\u2014") if contract else "\u2014"
        st.markdown(f"商户编号: {result.get('merchant_id', '')} | 合同编号: {cid}")

        st.markdown("**合同计费条款**")
        if contract:
            eff = contract.get("effective_date", "")
            end = contract.get("end_date", "")
            dl = contract.get("submit_deadline_day", 5)
            st.markdown(f"合同有效期: <span style='white-space:nowrap'>{eff} 至 {end}</span>", unsafe_allow_html=True)
            st.markdown(f"租金提交截止日: 每月 {dl} 日（宽限至 {dl + 3} 日）", unsafe_allow_html=True)
            if contract_type == "fixed":
                st.markdown(f"计费方式: 固定租金 \u00a5{contract.get('fixed_amount', 0):,.2f}/月")
            elif contract_type == "commission":
                st.markdown(f"计费方式: 营业额提成 {contract.get('commission_rate', 0) * 100:.2f}%")
            elif contract_type == "hybrid":
                mg = contract.get("min_guarantee", 0)
                rt = contract.get("commission_rate", 0)
                th = round(mg / rt, 2) if rt > 0 else 0
                st.markdown(f"计费方式: 保底 \u00a5{mg:,.2f}/月 或 营业额提成 {rt * 100:.2f}%（达标线 \u00a5{th:,.2f}），取较高值")
        else:
            st.caption("合同数据不可用")

        st.markdown("**当月营收数据**")
        if revenue_data:
            sd = revenue_data.get("submit_date", "")
            st.markdown(f"""
            申报营业额: ¥{revenue_data.get('reported_revenue', 0):,.2f}
            提交日期: <span style="white-space:nowrap">{sd}</span>
            """, unsafe_allow_html=True)
        else:
            st.caption("当月营收数据不可用")

    # 三、稽核结果
    # =========================================================================
    with st.expander("稽核结果", expanded=True):
        steps = audit_result.get("steps", [])
        date_status = "\u2014"
        amount_status = "\u2014"
        for step in steps:
            if step.get("step_name") == "提交日期校验":
                date_status = "正常" if step.get("status") == "success" else "逾期"
            if step.get("step_name") == "金额校验":
                amount_status = "正常" if step.get("status") == "success" else "异常"
        if amount_status == "\u2014":
            amount_status = "异常" if any(i["issue_type"] == "amount_mismatch" for i in issues) else "正常"
        col1, col2 = st.columns(2)
        dc = "#2e7d32" if date_status == "正常" else "#d32f2f"
        ac = "#2e7d32" if amount_status == "正常" else "#d32f2f"
        with col1:
            sd = revenue_data.get("submit_date", "\u2014") if revenue_data else "\u2014"
            st.markdown(f"**日期校验**<br/>提交: <span style='white-space:nowrap'>{sd}</span><br/>结果: <span style='color:{dc};font-weight:600;'>{date_status}</span>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**金额校验**<br/>结果: <span style='color:{ac};font-weight:600;'>{amount_status}</span>", unsafe_allow_html=True)

    # =========================================================================
    # 四、异常明细（含判定依据与计算过程）
    # =========================================================================
    has_issues = len(issues) > 0
    with st.expander("异常明细", expanded=has_issues):
        if not has_issues:
            st.markdown('<span style="color:#2e7d32;">正常</span> 所有校验通过', unsafe_allow_html=True)
        else:
            for i, issue in enumerate(issues, 1):
                sev = issue.get("severity", "medium")
                cid = issue.get("clause_id", "")
                itype = issue.get("issue_type", "")
                bc = {"high": "#d32f2f", "medium": "#e65100", "low": "#1565c0"}
                sl = {"high": "严重", "medium": "一般", "low": "提示"}
                il = {"amount_mismatch": "金额不匹配", "late_submission": "逾期提交"}

                clause_html = ""
                if cid and itype == "amount_mismatch":
                    try:
                        from src.retriever import get_clause_by_id
                        cd = get_clause_by_id(cid)
                        if cd:
                            raw = cd.get("description", "")
                            clause_html = f"""<details style="margin:0.5rem 0;font-size:0.9rem;">
<summary style="cursor:pointer;color:#1565c0;">查看合同条款原文 \u2014 {cid}</summary>
<div style="margin-top:0.25rem;padding:0.5rem;background:#f8f9fa;border-radius:4px;word-break:keep-all;">{raw}</div>
</details>"""
                    except Exception:
                        pass

                calc_html = ""
                if itype == "amount_mismatch" and revenue_data:
                    ev = issue.get("expected_value", 0)
                    av = issue.get("actual_value", 0)
                    rr = revenue_data.get("reported_revenue", 0)
                    diff = av - ev
                    if contract_type == "fixed":
                        formula = f"应缴 = 固定月租 \u00a5{ev:,.2f}"
                    elif contract_type == "commission":
                        rt = contract.get('commission_rate', 0)
                        formula = f"应缴 = 营业额 \u00a5{rr:,.2f} x {rt*100:.2f}% = \u00a5{ev:,.2f}"
                    else:
                        mg = contract.get("min_guarantee", 0)
                        rt = contract.get("commission_rate", 0)
                        cp = round(rr * rt, 2)
                        formula = f"应缴 = max(保底 \u00a5{mg:,.2f}, \u00a5{rr:,.2f} x {rt*100:.2f}% = \u00a5{cp:,.2f}) = \u00a5{ev:,.2f}"
                    calc_html = f"""<div style="margin:0.5rem 0;padding:0.5rem;background:#f8f9fa;border-radius:4px;font-size:0.9rem;">
<div style="margin-bottom:0.25rem;color:#616161;">{formula}</div>
<div>营业额 <strong>\u00a5{rr:,.2f}</strong>
<span style="color:#bdbdbd;margin:0 0.5rem;">|</span>
应缴 <strong>\u00a5{ev:,.2f}</strong>
<span style="color:#bdbdbd;margin:0 0.5rem;">|</span>
实缴 <strong>\u00a5{av:,.2f}</strong>
<span style="color:#bdbdbd;margin:0 0.5rem;">|</span>
<span style="font-weight:600;color:{"#d32f2f" if diff < 0 else "#2e7d32"};">差额 \u00a5{diff:+,.2f}</span>
</div></div>"""

                elif itype == "late_submission" and revenue_data:
                    try:
                        sdt = datetime.strptime(revenue_data.get("submit_date", ""), "%Y-%m-%d").date()
                        ys, ms = month.split("-")
                        by, bm = int(ys), int(ms)
                        ny = by + (1 if bm == 12 else 0)
                        nm = 1 if bm == 12 else bm + 1
                        dl = contract.get("submit_deadline_day", 5) + 3
                        dld = date(ny, nm, dl)
                        od = (sdt - dld).days
                        odt = f"逾期 {od} 天" if od > 0 else "宽限期内"
                    except Exception:
                        dld = "\u2014"
                        odt = "\u2014"
                    calc_html = f"""<div style="margin:0.5rem 0;padding:0.5rem;background:#f8f9fa;border-radius:4px;font-size:0.9rem;">
截止日（含宽限）<strong>{dld}</strong>
<span style="color:#bdbdbd;margin:0 0.5rem;">|</span>
实际提交 <strong>{revenue_data.get("submit_date", "")}</strong>
<span style="color:#bdbdbd;margin:0 0.5rem;">|</span>
<span style="color:#d32f2f;font-weight:600;">{odt}</span>
</div>"""

                st.markdown(f"""<div style="border-left:4px solid {bc.get(sev, '#d32f2f')};padding:0.25rem 1rem;margin-bottom:0.75rem;">
<div><span style="color:{bc.get(sev, '#d32f2f')};font-weight:600;">{sl.get(sev, '')}</span>
&nbsp;异常 #{i} \u2014 {il.get(itype, itype)}</div>
{clause_html}
{calc_html}
</div>""", unsafe_allow_html=True)

    # =========================================================================
    # 五、分析报告（默认展开）
    # =========================================================================
    with st.expander("分析报告", expanded=True):
        if mode == "deep" and report:
            summary = report.get("summary", "")
            suggestion = report.get("suggestion", "")
            if summary:
                st.markdown(f"**结论概述**\n\n{summary}")
            if suggestion:
                st.markdown(f"**处理建议**\n\n{suggestion}")
        else:
            # 标准模式：使用规则引擎的原始摘要 + 金额明细
            std_status = audit_result.get("status", "unknown")
            std_issue_count = len(issues)

            # 构建含金额细节的概述
            std_summary_parts = []
            for iss in issues:
                if iss.get("issue_type") == "amount_mismatch":
                    ev = iss.get("expected_value", 0)
                    av = iss.get("actual_value", 0)
                    diff = av - ev
                    std_summary_parts.append(f"应缴¥{ev:,.2f}，实缴¥{av:,.2f}，差额¥{diff:+,.2f}")
                elif iss.get("issue_type") == "late_submission":
                    std_summary_parts.append(f"报表逾期提交")
            amount_detail = "；".join(std_summary_parts)

            raw_summary_text = audit_result.get("summary", f"商户{result.get('merchant_id', '')} {month}稽核完成。")
            base_summary = raw_summary_text.replace("amount_mismatch", "金额不匹配").replace("late_submission", "逾期提交")
            if amount_detail:
                st.markdown(f"**结论概述**\n\n{base_summary}\n\n{amount_detail}")
            else:
                st.markdown(f"**结论概述**\n\n{base_summary}")

            if std_status == "abnormal":
                st.markdown(f"**处理建议**\n\n稽核发现{std_issue_count}项异常，建议查看异常明细并核实相关数据。")
            elif std_status == "normal":
                st.markdown(f"**处理建议**\n\n所有校验通过，无需处理。")
            else:
                st.markdown(f"**处理建议**\n\n稽核过程中出现数据缺失，建议人工核查。")
# ========== 导出按钮 ==========
    col_a, col_b, _ = st.columns([1, 1, 4])
    with col_a:
        report_json = json.dumps(result, ensure_ascii=False, indent=2)
        st.download_button(
            label="导出报告 (JSON)",
            data=report_json,
            file_name=f"audit_{result.get('merchant_id', '')}_{month.replace('-', '')}.json",
            mime="application/json",
        )
    with col_b:
        # 简单的文本报告
        # 报告编号
        report_id = f"AUD-{month.replace('-', '')}-{result.get('merchant_id', '')}-001"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode_label = "深度稽核" if mode == "deep" else "标准稽核"
        status_text = {"normal": "正常", "abnormal": "异常", "error": "待核查"}.get(status, "待核查")

        sev_zh = {"high": "严重", "medium": "一般", "low": "提示"}
        ctype_full = {"fixed": "固定租金", "commission": "营业额提成", "hybrid": "保底加提成"}

        text_lines = [
            "=" * 80,
            "            机场非航收入智能稽核报告",
            "=" * 80,
            "",
            f"报告编号：{report_id}",
            f"生成时间：{now_str}",
            f"稽核模式：{mode_label}",
            "",
            "=" * 80,
            "一、基本信息",
            "=" * 80,
            "",
            f"商户名称：{merchant_name}",
            f"商户编号：{result.get('merchant_id', '')}",
            f"合同编号：{contract.get('contract_id', '—') if contract else '—'}",
            f"合同类型：{ctype_full.get(contract_type, contract_type)}",
            f"合同有效期：{contract.get('effective_date', '') if contract else '—'} 至 {contract.get('end_date', '') if contract else '—'}",
            f"稽核期间：{month}",
            "",
            "=" * 80,
            "二、稽核结论",
            "=" * 80,
            "",
        ]

        if status == "abnormal":
            text_lines.append(f"稽核结果：{status_text}，共发现 {len(issues)} 项问题。")
        else:
            text_lines.append(f"稽核结果：{status_text}，未发现异常。")

        text_lines.extend([
            "",
            "=" * 80,
            "三、稽核项目判定",
            "=" * 80,
            "",
        ])

        # 金额比对
        steps = audit_result.get("steps", [])
        amount_step = None
        date_step = None
        for s in steps:
            if s.get("step_name") == "金额校验":
                amount_step = s
            if s.get("step_name") == "提交日期校验":
                date_step = s

        rev = revenue_data.get("reported_revenue", 0) if revenue_data else 0
        paid = revenue_data.get("paid_amount", 0) if revenue_data else 0

        text_lines.append("（一）金额比对")
        amount_clause_id = f"{contract.get('contract_id', 'CTR')}-clause-001" if contract else "\u2014"
        if amount_step:
            expected = amount_step.get("data", {}).get("expected_amount", 0)
            diff = paid - expected
            if amount_step.get("status") == "success":
                text_lines.append("判定结果：通过。")
            else:
                text_lines.append("判定结果：未通过。")
            text_lines.append(f"应缴金额{expected:,.2f}元，实缴金额{paid:,.2f}元，差额{diff:+,.2f}元。")
            text_lines.append(f"关联条款：{amount_clause_id}")
            clause_raw = ""
            try:
                from src.retriever import get_clause_by_id
                cd = get_clause_by_id(amount_clause_id)
                if cd:
                    clause_raw = cd.get("description", "")
            except Exception:
                pass
            if clause_raw:
                text_lines.append(f"条款原文：{clause_raw}")

            # 计算公式
            if contract_type == "fixed":
                fixed_amt = contract.get("fixed_amount", 0) if contract else 0
                text_lines.append(f"计费方式为固定租金，月租金{fixed_amt:,.2f}元。")
            elif contract_type == "commission":
                rate = contract.get("commission_rate", 0) * 100 if contract else 0
                text_lines.append(f"计费方式为营业额提成，提成比例为{rate:.2f}%。")
                text_lines.append(f"计算过程：申报营业额{rev:,.2f}元乘以{rate:.2f}%，得出{expected:,.2f}元。")
            elif contract_type == "hybrid":
                mg = contract.get("min_guarantee", 0) if contract else 0
                rate = contract.get("commission_rate", 0) * 100 if contract else 0
                cp = round(rev * (rate / 100), 2)
                text_lines.append("计费方式为保底加提成，取保底额与营业额提成的较高值。")
                text_lines.append(f"计算过程：以保底额{mg:,.2f}元为下限，申报营业额{rev:,.2f}元乘以提成比例{rate:.2f}%，"
                                  f"得出{cp:,.2f}元。两者比较后取{expected:,.2f}元。")
        else:
            text_lines.append("判定结果：无法判定（营收数据不可用）。")

        text_lines.append("")
        text_lines.append("（二）提交日期检查")
        date_clause_id = f"{contract.get('contract_id', 'CTR')}-clause-002" if contract else "\u2014"
        text_lines.append(f"关联条款：{date_clause_id}")
        clause_raw2 = ""
        try:
            from src.retriever import get_clause_by_id
            cd2 = get_clause_by_id(date_clause_id)
            if cd2:
                clause_raw2 = cd2.get("description", "")
        except Exception:
            pass
        if clause_raw2:
            text_lines.append(f"条款原文：{clause_raw2}")

        if revenue_data:
            sd = revenue_data.get("submit_date", "")
            dl = contract.get("submit_deadline_day", 5) if contract else 5
            # 简单计算逾期
            from datetime import datetime as ddt
            try:
                sdt = ddt.strptime(sd, "%Y-%m-%d").date()
                ys, ms = month.split("-")
                by, bm = int(ys), int(ms)
                ny = by + (1 if bm == 12 else 0)
                nm = 1 if bm == 12 else bm + 1
                dld = ddt(ny, nm, dl + 3).date()
                od = (sdt - dld).days
            except Exception:
                od = None
        else:
            sd = "—"
            od = None

        if date_step and date_step.get("status") != "success":
            text_lines.append("判定结果：未通过。")
            text_lines.append(f"合同约定提交截止日为每月{dl}日（宽限至{dl+3}日），"
                              f"实际提交日期为{sd}，逾期{od}日。" if od and od > 0 else
                              f"合同约定提交截止日为每月{dl}日（宽限至{dl+3}日），实际提交日期为{sd}，逾期{od}日。")
        else:
            text_lines.append("判定结果：通过。")
            text_lines.append(f"实际提交日期为{sd}，在合同约定截止日内。" if sd != "—" else "提交日期数据不可用。")

        # 异常明细
        if issues:
            text_lines.extend([
                "",
                "=" * 80,
                "四、异常明细",
                "=" * 80,
                "",
            ])
            for idx, issue in enumerate(issues, 1):
                sev = issue.get("severity", "medium")
                itype = issue.get("issue_type", "")
                cid = issue.get("clause_id", "")
                type_zh = {"amount_mismatch": "金额不符", "late_submission": "报表逾期提交"}

                text_lines.append(f"异常{idx}：{type_zh.get(itype, itype)}")
                text_lines.append(f"严重程度：{sev_zh.get(sev, sev)}")
                text_lines.append("")

                # 条款原文
                clause_raw = ""
                if cid:
                    try:
                        from src.retriever import get_clause_by_id
                        cd = get_clause_by_id(cid)
                        if cd:
                            clause_raw = cd.get("description", "")
                    except Exception:
                        pass

                if clause_raw:
                    text_lines.append(f"依据合同{cid}所在条款，条款原文如下：")
                    text_lines.append(f'"{clause_raw}"')
                    text_lines.append("")

                if itype == "amount_mismatch" and revenue_data:
                    ev = issue.get("expected_value", 0)
                    av = issue.get("actual_value", 0)
                    rr = revenue_data.get("reported_revenue", 0)
                    diff = av - ev

                    text_lines.append("计算过程：")
                    text_lines.append(f"1. 申报营业额：{rr:,.2f}元")

                    if contract_type == "commission":
                        rt = contract.get("commission_rate", 0) * 100 if contract else 0
                        text_lines.append(f"2. 按提成比例计算：{rr:,.2f}元乘以{rt:.2f}%，得出{ev:,.2f}元")
                        text_lines.append(f"3. 应缴金额：{ev:,.2f}元")
                    elif contract_type == "hybrid":
                        mg = contract.get("min_guarantee", 0) if contract else 0
                        rt = contract.get("commission_rate", 0) * 100 if contract else 0
                        cp = round(rr * rt / 100, 2)
                        text_lines.append(f"2. 按提成比例计算：{rr:,.2f}元乘以{rt:.2f}%，得出{cp:,.2f}元")
                        text_lines.append(f"3. 与保底额比较：{cp:,.2f}元{'高于' if cp > mg else '低于'}保底额{mg:,.2f}元")
                        text_lines.append(f"4. 应缴金额：{ev:,.2f}元（取较高值）")
                    else:
                        text_lines.append(f"2. 应缴金额：{ev:,.2f}元（固定租金）")

                    text_lines.append(f"{'5' if contract_type in ('hybrid','commission') else '3'}. 实缴金额：{av:,.2f}元")
                    text_lines.append(f"{'6' if contract_type in ('hybrid','commission') else '4'}. 差额：实缴较应缴{'少' if diff < 0 else '多'}{abs(diff):,.2f}元")

                elif itype == "late_submission" and revenue_data:
                    sd = revenue_data.get("submit_date", "")
                    text_lines.append(f"合同约定提交截止日为每月{dl}日（宽限至{dl+3}日）。")
                    text_lines.append(f"该商户{month}报表实际提交日期为{sd}，逾期{od}日。" if od and od > 0 else
                                      f"该商户{month}报表实际提交日期为{sd}，逾期{od}日。")
                    text_lines.append("根据合同约定，应按日收取滞纳金。")

                text_lines.append("")

        # 分析建议
        rpt = result.get("report", {})
        text_lines.extend([
            "=" * 80,
            "五、分析建议",
            "=" * 80,
            "",
        ])
        if mode == "deep" and rpt:
            summary = rpt.get("summary", "")
            suggestion = rpt.get("suggestion", "")
            if summary:
                text_lines.append("结论概述：")
                text_lines.append(summary)
                text_lines.append("")
            if suggestion:
                text_lines.append("处理建议：")
                for line in suggestion.split("\\n"):
                    if line.strip():
                        text_lines.append(line.strip())
                text_lines.append("")
        else:
            # 标准模式：使用规则引擎摘要 + 金额明细
            std_status = audit_result.get("status", "unknown")
            std_detail_parts = []
            for iss in issues:
                if iss.get("issue_type") == "amount_mismatch":
                    ev = iss.get("expected_value", 0)
                    av = iss.get("actual_value", 0)
                    diff = av - ev
                    std_detail_parts.append(f"应缴{ev:,.2f}元，实缴{av:,.2f}元，差额{diff:+,.2f}元")
                elif iss.get("issue_type") == "late_submission":
                    std_detail_parts.append("报表逾期提交")
            amount_detail = "\uff1b".join(std_detail_parts)
            raw_summary_text = audit_result.get("summary", f"商户{result.get('merchant_id', '')}{month}稽核完成。")
            std_summary = raw_summary_text.replace("amount_mismatch", "金额不匹配").replace("late_submission", "逾期提交")
            text_lines.append("结论概述：")
            text_lines.append(std_summary)
            if amount_detail:
                text_lines.append(amount_detail)
            text_lines.append("")
            text_lines.append("处理建议：")
            if std_status == "abnormal":
                text_lines.append(f"稽核发现{len(issues)}项异常，建议查看异常明细并核实相关数据。")
            elif std_status == "normal":
                text_lines.append("所有校验通过，无需处理。")
            else:
                text_lines.append("稽核过程中出现数据缺失，建议人工核查。")
            text_lines.append("")

        text_report = "\n".join(text_lines)
        st.download_button(
            label="导出报告 (TXT)",
            data=text_report,
            file_name=f"audit_{result.get('merchant_id', '')}_{month.replace('-', '')}.txt",
            mime="text/plain",
        )

    # ========== 耗时信息 ==========
    st.divider()
    trace = result.get("trace", [])
    trace_lines = []

    # 从 trace 提取各步骤耗时（如果有 duration_ms 字段）
    # 当前 trace 没有 duration_ms，所以用总耗时替代
    trace_lines.append(f"总耗时 {elapsed:.1f} 秒")

    # 统计 LLM 调用次数
    llm_count = sum(1 for t in trace if t.get("node") in ("planner", "report"))
    if llm_count > 0:
        trace_lines.append(f"LLM 调用 {llm_count} 次")

    # 模式
    mode_label = "深度稽核" if mode == "deep" else "标准稽核"
    trace_lines.append(f"稽核模式: {mode_label}")

    # 完成时间
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trace_lines.append(f"完成时间: {now_str}")

    st.caption(" | ".join(trace_lines))


# ===========================================================================
# 八、主页面
# ===========================================================================

def main():
    """主页面"""
    init_session_state()
    inject_css()

    st.set_page_config(
        page_title="机场非航收入智能稽核系统",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("机场非航收入智能稽核系统")
    st.markdown(
        '<span style="color:#616161;font-size:0.95rem;">'
        "基于确定性规则引擎与Agentic RAG的自动化稽核平台"
        '</span>',
        unsafe_allow_html=True,
    )

    # 侧边栏
    merchant_id, month, run_clicked, is_deep_mode, nl_input, is_nl_active = render_sidebar()

    # ===== 稽核执行 =====
    if run_clicked:

        # ---- 自然语言路径：解析失败时提前退出，不进入 spinner ----
        nl_failed = False
        if is_nl_active and nl_input.strip():
            parsed = parse_natural_language(nl_input)

            if parsed and parsed.get("merchant_id") and parsed.get("month") and not str(parsed.get("month", "")).startswith("multi"):
                # 解析成功：提取参数，进入 spinner
                merchant_id = parsed["merchant_id"]
                month = parsed["month"]
                name = parsed.get("merchant_name", merchant_id)
                st.caption(f"已识别: {name} / {month}")
            else:
                nl_failed = True
                # 各类失败提示（无 spinner）
                if not parsed:
                    st.warning("暂时无法理解您的查询，请尝试使用下拉选择方式")
                    st.caption("例如：查看云松咖啡3月的租金缴纳情况")
                else:
                    etype = parsed.get("error_type", "")
                    ereason = parsed.get("error_reason", "")

                    if etype == "empty_input" or len(nl_input.strip()) < 3:
                        st.info("请输入查询内容")
                        st.caption('例如：查看云松咖啡3月的租金缴纳情况、查一下麦香餐厅有没有逾期')
                    elif etype == "multi_month":
                        msg = "当前每次仅支持查询一个月份的稽核结果。您的查询中包含多个月份，无法同时执行。"
                        if parsed.get("merchant_name"):
                            msg += f"\n\n已为您识别到商户“{parsed['merchant_name']}”，您可以分别查询每个月份，或使用左侧下拉列表逐个选择月份。"
                        st.warning(msg)
                    elif etype == "merchant_not_found":
                        merchants = load_merchants()
                        names = [f"{m['merchant_name']}（{m['merchant_id']}）" for m in merchants[:10]]
                        msg = ereason if ereason else "未找到该商户的稽核记录"
                        msg += "\n\n系统中现有的商户包括：\n" + "、".join(names) + "……\n\n请确认商户名称是否正确，或在左侧下拉列表中直接选择。"
                        st.warning(msg)
                    elif etype == "merchant_ambiguous":
                        candidates = parsed.get("merchant_candidates", "")
                        msg = f"查询“{parsed.get('merchant_name', '')}”匹配到多个商户"
                        if candidates:
                            msg += f"：{candidates}"
                        msg += "\n\n请在左侧下拉列表中直接选择对应的商户。"
                        st.warning(msg)
                    elif etype == "no_match":
                        st.warning("未能识别出明确的商户名称和月份")
                        st.caption("请尝试使用以下格式重新输入：\n查看云松咖啡3月的租金缴纳情况\n查一下麦香餐厅有没有逾期\n星野小馆1月的报表提交日期")
                    else:
                        st.warning(ereason if ereason else "暂时无法处理您的查询，请重试")
                        st.info("您也可以使用左侧的下拉选择方式直接选择商户和月份")
                render_history_table()

        # ---- 进入 spinner 实际执行稽核 ----
        if nl_failed:
            st.stop()
        else:
            with st.spinner("正在执行稽核，请稍候..."):
                start = time.time()
                if is_nl_active:
                    mode = "deep"
                    result = run_audit_deep(merchant_id, month)
                elif is_deep_mode:
                    mode = "deep"
                    result = run_audit_deep(merchant_id, month)
                else:
                    mode = "standard"
                    result = run_audit_standard(merchant_id, month)
                elapsed = time.time() - start

        if result and not nl_failed:
            append_history(result, elapsed)
            render_audit_result(result, elapsed, mode)
        else:
            st.error("稽核服务暂不可用，请稍后重试或联系系统管理员")
            render_history_table()

    else:
        # ===== 默认状态：稽核工作台 =====
        render_history_table()


if __name__ == "__main__":
    main()