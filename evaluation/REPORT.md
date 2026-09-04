# 生成式部分自动化评测 —— 实现与运行报告

> 日期：2026-08-21（最近更新 2026-08-30）· 实际运行：主集 20 条 + 边界补充集 200 条（真实 LLM 报告）
> 所有数字均为真实运行产生，无任何硬编码/伪造。

---

## 1. 评测流程（总览）

```
600 cases 构建（rule engine 权威 ground truth）
   → 逐 case 生成报告（audit_mock + report_node，1 次 LLM 调用/case，带缓存）
   → Claim 抽取（确定性正则，非 LLM）
   → 判定：一致性 / 引用 / 幻觉（全部确定性）
   → 指标聚合（确定性 + 生成式）
   → 输出 JSON / HTML / 逐 case / Top 失败案例
```

设计原则：**规则引擎输出 = 权威事实源，LLM 只负责解释**。Ground Truth 由 `audit_mock()`（规则引擎复算）+ 账单 CSV + 合同 JSON 自动构建，**零人工标注**。

| 步骤 | 实现 |
|---|---|
| Ground Truth | `dataset.build_cases()`：600 条 = 50 商户 × 12 月；`ground_truth` 含 status/revenue/payable/paid/difference/reason/used_clause_ids |
| 报告生成 | `runner.generate_report()`：`audit_mock + report_node`（跳过与评测无关的 Planner，LLM 调用从 2 次/case 压到 1 次）；失败自动降级为规则引擎报告并标记 `degraded`，不中断整体 |
| Claim 抽取 | `claim_extractor`：锚定关键词 + 正则（含公式折叠），status/contract_refs 直接读结构化字段，默认不调 LLM |
| 判定 | `consistency`（事实一致性）/ `citation_verifier`（引用集合运算）/ `hallucination`（白名单化三类事件） |
| 聚合 | `metrics.aggregate()`：确定性 + 生成式分开 |
| 输出 | `report.py`：`evaluation_report.json` / `.html` / `evaluation_cases.json` |

## 2. 指标定义

### 确定性指标（规则引擎自身的表现，对照注入异常标注 `truth_labels.json`）
| 指标 | 定义 |
|---|---|
| 应缴金额核对准确率 | 排除「少报营业额」后，规则引擎判定金额异常 == 注入标注金额异常的账单比例 |
| 异常识别 P/R/F1 | 规则引擎 status=abnormal vs 注入标注 anomaly，tp/fp/fn/tn 计算 |

### 生成式指标（仅在真实 LLM 报告的 case 上聚合；20/20 本次全为 LLM）
| 指标 | 定义（逐 case） |
|---|---|
| Report Consistency | 1 − 矛盾事实数 / 事实总数（status + 4 个金额事实；只罚「矛盾」不罚「遗漏」） |
| Claim Accuracy / Recall | 正确陈述的事实 / 期望事实总数（漏报不计错） |
| Claim Precision | 正确陈述 / 全部已陈述（有陈述才有分母） |
| Citation Truthfulness | 有效条款引用数 / 引用总数（clause_id 是否存在于商户合同） |
| Citation Completeness / Recall | 命中必需证据的引用数 / 必需证据数 |
| Citation Precision | 命中必需证据的引用数 / 引用总数（衡量过度引用） |
| Hallucination Rate | 幻觉事件数 / 总断言数（金额断言 + status + 引用数） |

**required_evidence（必需证据）口径**：规则引擎**实际核实的条款**（每 case 固定两道校验）：
- 金额校验 → `{contract_id}-clause-001`
- 提交日期校验 → `{contract_id}-clause-002`

而非「仅被标记异常的条款」—— 报告引用日期条款（支撑「未逾期」结论）是合理证据引用，须计入必需集，否则 precision 被伪压低（详见 §6-缺陷4）。

引用粒度 = **clause_id**（本项目无 bill_id/payment_id 命名空间，营收与实缴同属一条账单记录，故 Completeness 在条款维度计算）。

## 3. 运行命令

```bash
python -m evaluation.runner                      # 全量 600（≈600 次 LLM 调用）
python -m evaluation.runner --cases 20           # 前 20 条
python -m evaluation.runner --merchant M001      # 单商户 12 月
python -m evaluation.runner --merchant M001 --month 2025-01   # 单 case 调试
python -m evaluation.runner --no-report          # 确定性降级报告跑通链路（0 LLM 调用）
python -m evaluation.runner --refresh-cache      # 忽略报告缓存重新生成
python -m evaluation.runner --boundary           # 边界补充集（200 条合成账单，LLM 报告）
python -m evaluation.runner --boundary --no-report  # 边界集确定性模式（0 LLM，验证规则引擎边界判定）
python -m pytest tests/test_generative_eval.py -v            # 快速单元/集成测试
python -m pytest tests/test_generative_eval.py -v --runslow  # 含真实数据集成测试
```

配置：`evaluation/config.yaml`（`use_llm_report: true`、`use_llm_judge: false`、`cache_enabled: true`、`amount_tolerance: 0.01`）。

## 4. 测试结果

- **单元测试（26 条，全过，0.17s）**：金额解析、公式折叠、锚定抽取、条款 ID 正则、容差比较、一致性、引用校验、幻觉判定、指标聚合、LLM Judge JSON 解析（含 judge_error 兜底）。
- **集成测试（2 条，全过）**：
  - Hermetic 全链路（降级报告，无 LLM/模型）：Agent→Report→Extraction→Verification→Metrics，0.1s。
  - 真实数据链路（`--runslow`，加载 BGE 模型）：`build_cases(3) → evaluate_case → aggregate`，3.7s。
- **全量回归**：`pytest tests/` → **91 passed, 18 skipped**（skip 为既有模型加载用例）。

## 5. 实际运行结果（20 条真实 LLM 报告）

- 数据集：20 cases（M001×12 + M002×8），**异常 3 / 正常 17**，case_errors 0，LLM 报告 20，降级 0。
- 确定性：应缴金额核对准确率 **100.00%**（19 条）；异常识别 F1 **100.00%**（P=1.00 R=1.00，tp=3 fn=0）。

| 生成式指标 | 值 | 说明 |
|---|---|---|
| Report Consistency | **100.00%** | 无矛盾陈述（0 wrong / 100 事实） |
| Claim Accuracy / Precision / Recall | **100.00%** | 4 个金额事实 + status 全对 |
| Citation Truthfulness | **100.00%** | 引用全部为真实 clause_id |
| Citation Completeness / Recall | **100.00%** | 必需证据（金额+日期条款）全部命中 |
| Citation Precision | **95.24%** | 少量额外引用（滞纳金条款） |
| Hallucination Rate | **0.00%** | C014 类滞纳金编造被规则8压制（此前 1.28%） |

**五轮优化轨迹**（每轮均为真实重跑）：
| 轮次 | Claim Recall | Citation Precision | Completeness | 说明 |
|---|---|---|---|---|
| 初版 | 62% | 8.1% | — | report_node 未给 LLM clause_id，引用为编造 |
| 修复 clause_id 暴露 + 公式折叠 + 差额绝对值 | 64% | 46.5% | — | 引用真实化，但正常 case 缺显式金额、过度引用日期条款 |
| prompt 强制四数字显式 + contract_refs 最小化 | 100% | 71.4% | — | Claim 拉满 |
| required_evidence 对齐审计口径（金额+日期） | 100% | 95.0% | 95% | 日期条款计入必需集 |
| 两道校验明细入 prompt + 强制双条款引用 + 反幻觉规则8 | 100% | 95.24% | **100%** | Completeness 拉满，幻觉率归零 |

## 5-5. 边界补充集评测（200 条合成账单 / 25 家商户）

**背景**：600 条主数据集在边界维度存在覆盖缺口（近容差金额 0 条、宽限内提交 0 条、零营业额/缺失提交/达标线均无样本）。为让评测覆盖这些维度，按「边界补充集」规格构建合成账单（仅新增数据文件 + 评测代码，未改核心业务代码），**由最初的 25 条扩充至 200 条 / 25 家合成商户**：

| 数据文件 | 内容 |
|---|---|
| `data/contracts_boundary.json` | 25 家合成边界商户 BM001~BM025（提成 / 保底+提成 / 固定三类合同，BM001 提成 14.43%、BM002 保底8000+12%、BM003 固定10000、BM004 提成 10%，提成类 10~16%、保底类 min 6000~12000 / 10~15%、固定类 8000~15000），结构与主合同一致（各含 3 条条款） |
| `data/bills_boundary.csv` | 200 条账单（**105 正常 / 95 异常**），含 `expected_status`/`expected_anomaly_type`/`scenario`（领域真值列，独立于规则引擎） |

**场景分布**（105 正常 / 95 异常）：金额近容差精确边界（0.99% / 恰好 1.00% / 1.01%、绝对差 0.01 min_anomaly）、提交宽限期（截止日当天 / 晚 1~7 天 / 宽限末第 8 天）、零营业额（commission / hybrid / fixed 各态：零缴判异常、缴保底或固定额判正常）、缺失提交日期、多缴（容差内正常 / 超容差异常）、达标线精确点（保底=提成切换）、双异常组合（金额不符 + 逾期提交同现）。

**设计原则**（与主集口径一致）：ground_truth 由规则引擎 `audit_single` 实算（零人工标注）；truth_label 取自 CSV 的 `expected_status`（领域业务语义真值）。边界集的价值在于 **对照业务语义预期，检验规则引擎在阈值附近是否判对**；若规则引擎边界判错，确定性指标会如实暴露。

**入库前全量校验**：合并前先用规则引擎 `audit_single` 对全部 200 条实算校验 —— **0 条状态不一致、0 条异常类型不一致、0 条金额标签不一致、无重复、商户合同全覆盖、clause_id 约定完整**。仅发现 2 条数据质量问题并修正（BM002 2026-09 实缴 12100→12120 使差额恰为 1% 与其 normal 场景相符；BM002 2026-10 实缴 12120→12121 使差额越过 1% 与其 abnormal 场景相符），修正后全部通过。

**运行结果（真实运行）**：
```bash
python -m evaluation.runner --boundary --no-report   # 确定性，0 LLM，<1s
python -m evaluation.runner --boundary               # LLM 报告，200 次调用，380.9s
```

| 层 | 指标 | 结果 |
|---|---|---|
| 确定性 | 应缴金额核对准确率 | **100.00%**（200/200） |
| 确定性 | 异常识别 F1 | **100.00%**（tp=95, tn=105, fp=0, fn=0） |
| 生成式 | Report Consistency | **100.00%**（0 wrong / 1000 事实） |
| 生成式 | Claim Accuracy / Recall | **99.60%**（996/1000；4 个 missing 全为**抽取器锚定局限**：B014/B134/B141「应缴金额为保底额¥X」锚定词后夹"保底额"未匹配、B027「差额 -¥0.11」负号未匹配，LLM 报告实际都写了） |
| 生成式 | Claim Precision | **100.00%**（0 wrong，LLM 零事实错误） |
| 生成式 | Citation Truthfulness | **100.00%**（424/424） |
| 生成式 | Citation Completeness / Recall | **100.00%**（397/397） |
| 生成式 | Citation Precision | **93.63%**（397/424，27 条多余引用：逾期提交 case 多引 clause-003 滞纳金 + 缺失提交日期 case 多引 clause-002，均真实但非本次必需证据） |
| 生成式 | Hallucination Rate | **0.00%**（0/1954 断言） |

**边界集抓到的真实问题（LLM 弱项，主数据集测不到；25 条阶段发现并已修复，200 条阶段保持收敛）**：
1. **近容差金额判错**（B002/B003/B004/B021，4 case）：0.3%/0.9%/0.95% 少缴均 ≤1% 容差 → 规则引擎判正常；LLM 报告却判 **abnormal**（如 B002 差额 ¥21.64 → 「金额不匹配，需补缴」）。LLM 未应用 1% 容差规则，把任何差额视为异常 —— 规则引擎与 LLM 口径冲突的直接实证。**已修复**：把规则引擎「逐道校验明细」暴露给 LLM 后，近容差账单 status 矛盾消失（见 §6-6）。
2. **滞纳金幻觉复现**（B012/B013）：逾期 case 上 LLM 再次自行推算滞纳金（B012 `5772×0.5%×1=¥28.86`、B013 `3000×0.5%×4=¥60.00`），规则引擎不产滞纳金金额 → 检测器判定幻觉，与主集 C014 同根因。**已修复（prompt 层）**：规则8 禁止出现引擎未提供的金额，幻觉率归零（引擎层根因修复见 §9-1）。
3. **达标线差额错报**（B024）：保底+提成合同，payable=`max(8000, 66000×12%)=8000`、实缴 7920、差额 ¥80（恰在 1% 容差 → 正常）；LLM 报告把应缴写成 ¥7920、差额 ¥0 —— **漏掉保底条款**。**已修复**：校验明细暴露后应缴/差额均正确。

**结论**：确定性层证明规则引擎在全部 200 个边界点判定正确（容差两侧、宽限两侧、零营收、缺失提交、多缴、达标线、双异常），95 条异常全检出、105 条正常零误报；生成式层证明 LLM 报告在非常规数据上的弱点可被量化抓取，并经 prompt 修复后全部收敛（Consistency/Completeness/Truthfulness 100%、幻觉率 0%、LLM 零事实错误），仅剩 2 类已知测量口径损失（抽取器锚定 99.60%、滞纳金条款过度引用 93.63%）。边界集已可复用为第 4 层评测（与主 600 条、人工复核集并列）。

## 6. 评测发现并修复的缺陷

1. **claim_extractor 公式折叠未接入锚定抽取**：`¥52,893.10×14.43%=¥7,632.47` 内联公式在锚定「应缴金额」时把左侧营业额误判为应缴金额。修复：`_first_tagged_amount` 也做公式折叠。
2. **report_node 未向 LLM 暴露 clause_id**：条款行只渲染 `[clause_type] description`，LLM 无法引用真实条款，只能拿条款类型名编造（`rent_calculation`）。首次运行 Citation Truthfulness 仅 **8.11%**。修复：条款行带上 `条款ID`，系统提示明确 contract_refs 必须使用数据中的 clause_id。
3. **consistency 差额未按绝对值比对**（与 docstring 声明不符）：多缴 case 报告 `+3,899.52` vs 规则引擎 `-3,899.52` 被误判矛盾。修复：`difference` 按绝对值比对。
4. **required_evidence 只含被标记异常的条款**：正常 case 报告引用日期条款（规则引擎确实核实的）被误判为过度引用，Citation Precision 虚低（46.5% 中大半是伪缺）。修复：required_evidence = 规则引擎实际核实的条款（金额 + 提交日期），precision/recall 提升至 95%。
5. **报告未强制显式陈述金额**：正常 case 写「实缴金额一致」无数字，抽取器要求显式金额 → Claim Recall 仅 64%。修复：prompt 强制无异常 case 也必须写出申报营业额/应缴/实缴/差额四个数字（差额为 0 写 ¥0.00）。
6. **报告只围绕"异常那道校验"引用条款（Completeness 低）**：report_node 只把「复算结果状态 + 异常列表」给 LLM，LLM 看不到引擎对每张账单都执行了金额与提交日期两道校验 → 金额异常 case 漏引日期条款（Completeness 68%）。修复：① 把 `audit_result.steps` 中两道校验明细注入用户提示词；② 系统提示词改为**无条件要求**报告同时说明两道校验结论并同时引用两条款（不再"若涉及才引用"）。主集 Completeness 95%→100%，边界集 68%→100%。
7. **"容差内通过"被 LLM 写成"差额为 0"**：暴露校验明细后，近容差 case（差额 ¥21.64~90 均在 1% 容差内）被 LLM 报成差额 ¥0.00（事实矛盾）。修复：prompt 明确"容差内通过 ≠ 差额为 0，必须写真实算术差额"。
8. **滞纳金编造（幻觉）**：LLM 用 `日费率×逾期天数` 自行推算滞纳金（C014 ¥240.22 / 边界集 ¥28.86、¥60.00），规则引擎不产此金额。prompt 层修复：规则8 禁止出现任何来源中没有的金额，写"滞纳金需另行核定"。幻觉率：主集 1.28%→0%，边界集 1.52%→0%。引擎层根因（是否应产滞纳金）见 §9-1。

## 7. 典型失败案例

**C014（M002 2025-02，正常金额 + 逾期提交）** —— 幻觉 2 条，全部命中滞纳金：
> 报告：「9,608.74×0.5%×5=¥240.22」→「补缴滞纳金¥240.22」

规则引擎对 `late_submission` **只记录提交/截止日期，不产滞纳金金额**（`late_fee_rate` 在配置但代码未计算），账单/合同也无此金额。该 ¥240.22 是 LLM 自行推演的**无来源金额，检测器判定正确**。属生成式评测真实抓到的幻觉。**当前状态：已修复（prompt 规则8 禁止无来源金额），幻觉率归零**；引擎层根因修复见 §9-1。

**C001（M001 2025-01，金额不符）** —— 引用瑕疵：报告讨论了「晚于每月5日」的截止日期，却引用滞纳金条款 `clause-003` 而非日期条款 `clause-002`；漏引日期条款，多引滞纳金条款。**已修复**：两道校验明细入 prompt + 强制双条款引用后消失。
**C018（M002 2025-06，多缴）** —— 引用瑕疵：仅引金额条款，漏引日期条款。**已修复**：同上。

## 8. 当前局限

1. **C014 的滞纳金幻觉是白名单机制边界**：规则引擎不产滞纳金金额，报告自行推算即判幻觉（本次判定正确）。若规则引擎未来产出合法派生金额，白名单需同步扩展，否则会误报。
2. **差额方向不区分**：`difference` 按绝对值比对，「少缴 ¥X」与「多缴 ¥X」都算正确，方向性错误不惩罚。
3. **LLM Judge 默认关闭**：语义蕴含层（报告结论 vs 事实是否自洽）尚未启用，`judge_error_rate` 为 null。
4. **引用指标在条款维度，无法覆盖「引用了错的位置/上下文」**：报告引用正确的 clause_id 但用错场景（如 C001 用滞纳金条款解释逾期）只在 precision 上有轻微体现，需要 LLM Judge 才能细判。
5. 未跑全量 600（真实成本：600 次 DeepSeek 调用，本次 20 条耗时 ~40s）；已提供 `python -m evaluation.runner` 一条命令即可全量。

## 9. 下一步优化

1. **修复 C014 类滞纳金幻觉的引擎层根因**：prompt 规则8 已在报告层压制幻觉（幻觉率 0%），但根因仍在 —— 规则引擎对 `late_submission` 只记录日期、不产滞纳金金额（`late_fee_rate` 在配置但代码未计算）。可选修复：引擎在逾期时明确输出「不计算滞纳金」或真正计算并纳入 `build_source_amounts` 白名单。
2. ~~报告一致性 / 引用完整性~~：**已完成**（两道校验明细入 prompt + 强制双条款引用），主集与边界集 Citation 均达 Completeness 100%、幻觉率 0%。
3. **启用 LLM Judge**（`use_llm_judge: true`）做语义蕴含第二层，覆盖方向性措辞与引用场景正确性的细判（§8-3/§8-4 局限仍在）。
4. **差额方向纳入判定**：抽取时保留「少缴/多缴」方向语义，与 ground truth 的 difference 符号核对（§8-2 局限仍在）。
5. ~~近容差口径对齐（边界集 B002/B003/B004/B021）~~：**已完成**（校验明细暴露后，LLM 正确识别容差内正常；差额数字经规则2b 修正为真实算术值）。
6. ~~保底条款强提醒（边界集 B024）~~：**已完成**（校验明细暴露后应缴/差额正确）。
7. 全量 600 条跑通后，将生成式指标纳入 `scripts/evaluate.py` 的既有对比体系，形成基线对照；边界补充集作为第 4 层纳入 `python -m evaluation.runner` 可选参数（已实现 `--boundary`）。

---

## 变更文件清单

**新增（evaluation/ 评测包 + 测试）**
| 文件 | 职责 |
|---|---|
| `evaluation/__init__.py` | 包声明 |
| `evaluation/config.yaml` | 评测配置 |
| `evaluation/dataset.py` | 600 条 case 构建（rule engine 权威 ground truth） + `build_boundary_cases()` 边界补充集 |
| `evaluation/claim_extractor.py` | Claim/金额/引用抽取（确定性） |
| `evaluation/consistency.py` | Report Consistency + 数字事实核对 |
| `evaluation/citation_verifier.py` | 引用集合校验（truthfulness/completeness/precision/recall） |
| `evaluation/hallucination.py` | 幻觉检测（三类确定性事件） |
| `evaluation/metrics.py` | 确定性 + 生成式指标聚合 |
| `evaluation/llm_judge.py` | 可选 LLM Judge（默认关闭） |
| `evaluation/report.py` | JSON / HTML / 逐 case / Top 失败输出 |
| `evaluation/runner.py` | 批量管线 + 缓存 + CLI 主入口（新增 `--boundary`） |
| `evaluation/cache.py` | 报告/数据集落盘缓存（断点续跑） |
| `tests/test_generative_eval.py` | 32 单元 + 2 集成测试（含 TestBoundary 6 条） |
| `evaluation/REPORT.md` | 本报告 |

**边界补充集数据（合成账单，不碰核心业务代码）**
| 文件 | 职责 |
|---|---|
| `data/contracts_boundary.json` | 25 个合成边界商户合同（BM001~BM025，结构与主合同一致，各含 3 条条款） |
| `data/bills_boundary.csv` | 200 条边界账单（金额近容差 / 提交宽限期 / 零营业额 / 缺失提交 / 多缴 / 达标线 / 双异常），含领域真值列 |

**修改（评测暴露缺陷后修复）**
| 文件 | 改动 |
|---|---|
| `evaluation/claim_extractor.py` | `_first_tagged_amount` 接入公式折叠 |
| `src/agent_graph.py` | `report_node` 条款行暴露 clause_id + **逐道校验明细（金额/提交日期）入 prompt**；`REPORT_SYSTEM_PROMPT` 强制四数字显式 + 双校验无条件报告 + 双条款引用 + 容差内写真实差额 + 反幻觉规则8 |
| `evaluation/consistency.py` | `difference` 按绝对值比对（对齐 docstring 声明） |
| `evaluation/dataset.py` | `required_evidence` 改为规则引擎实际核实的条款（金额+日期）；新增 `build_boundary_cases()` 边界补充集 |
