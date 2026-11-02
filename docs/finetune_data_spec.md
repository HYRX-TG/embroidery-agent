# 知绣坊 · 微调数据规格说明（v2）

本文档定义 `data/finetune/` 四类微调数据的**用途、字段格式、数量目标、质量红线与验收标准**。
所有数据供 LlamaFactory QLoRA 微调 Qwen2.5 系列使用，经 `scripts/export_finetune.py` 导出为 alpaca 格式。

## 1. 微调目标

让基座模型学会三件事：
1. **行为规范**：知绣坊客服的角色边界（不报价、不承诺工期库存、门店专属信息转人工、温柔国风话术、标注来源）；
2. **行业表述**：刺绣工艺/针法/品类/保养/纹样寓意的准确、一致表述（与 `data/knowledge/` 对齐）；
3. **结构化能力**：从客户对话中提取定制需求表单 JSON、多轮追问中继承上文参数（省略句）。

## 2. 四类数据一览

| 文件 | 场景 | 字段 | 本批目标 | 累计目标 |
|---|---|---|---|---|
| raw_qa.jsonl | C端单轮知识问答 | instruction / answer | ≥100条 | 500条 |
| raw_dialogue.jsonl | C端多轮对话（定制追问/上下文） | conversations / system(可选) | ≥20组 | 200组 |
| raw_staff.jsonl | B端员工（答疑/复盘/文案/报价解释） | instruction / input(可选) / answer / system(可选) | ≥40条 | 300条 |
| raw_form.jsonl | 需求表单结构化输出 | instruction / form | ≥40条 | 200条 |

本批合计约 200+ 条高质量种子集；累计千条级通过 `scripts/generate_finetune.py`（模板+知识条目组合）扩充，人工抽检后合入。

## 3. 逐类字段规范

### 3.1 raw_qa.jsonl（C端单轮问答）

- 每行一个 JSON 对象：`{"instruction": "<客户问题>", "answer": "<客服回答>", "system": "<可选覆盖>"}`
- 内容覆盖：工艺对比、针法介绍、品类介绍、保养存放、纹样寓意、送礼/家装选购建议、定制引导、转人工话术。
- 答案约束：
  - 知识类必须与知识库口径一致，可写"参考《xxx》"类来源；
  - 涉及价格/工期/库存/优惠一律不回答具体数字，转人工话术；
  - 单条答案 40–200 字，语气温柔国风但不过度堆砌。

### 3.2 raw_dialogue.jsonl（C端多轮对话）

- 每行一个 JSON 对象：`{"conversations": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}, ...], "system": "<可选>"}`
- 覆盖场景：
  a) 定制需求追问循环（尺寸→工艺→用途→预算→时限→特殊要求，每轮只问1个）；
  b) 需求信息齐全后的【结构化需求表单】+【方案建议】+【待人工确认】输出；
  c) 省略句继承（"那只要5幅呢"承接上文尺寸/工艺/装裱）；
  d) 中途退出定制（"先看看"）；
  e) 多轮知识咨询。
- 组内轮数 2–8 轮，最后一轮必须是 assistant。

### 3.3 raw_staff.jsonl（B端员工）

- 每行一个 JSON 对象：`{"instruction": "<任务>", "input": "<可选材料，如客户对话>", "answer": "<员工结果>", "system": "<可选覆盖>"}`
- 覆盖场景：
  a) 行业知识答疑（含内部标准口径）；
  b) 客户对话复盘（提炼诉求/预算/成交概率/跟进建议）；
  c) 文案生成（朋友圈/口播/详情页/话术）；
  d) 报价逻辑解释（内部口径，禁止对外报价）；
  e) 售后处理话术。
- 红线：答案不得包含"给客户报价的具体数字"，复盘/报价解释标注"内部口径"。

### 3.4 raw_form.jsonl（需求表单结构化输出）

- 每行一个 JSON 对象：`{"instruction": "<客户对话（可含多轮）>", "form": {"pattern_theme":"","size":"","craft_type":"","quantity":"","usage":"","budget_range":"","deadline":"","special_requirements":"","core_preferences":""}}`
- form 字段与 `RequirementForm` 一一对应；未知字段填空字符串；
- instruction 至少包含 2 个以上可提取字段；覆盖：主题+尺寸、全字段齐备、省略句继承、口语化表达、含装裱/颜色等特殊要求。

## 4. 质量红线（写数据时逐条遵守）

1. 不编造知识：所有知识表述以 `data/knowledge/`、`data/team_kb/`、`data/internal/` 为口径；超出知识库的部分用常识但不虚构具体数字。
2. 不报价：C端一律"由顾问核算"，不给具体价格；员工侧报价只讲口径与区间逻辑，不给客户承诺。
3. 不承诺：工期、库存、优惠一律转人工。
4. 来源可追溯：知识类答案标注"参考《来源》"或来源编号。
5. 输出纪律：form 类只输出 JSON；话术类不用 JSON。
6. 多样性：同一知识点至少 2 种问法；避免连续重复句式。

## 5. 验收标准（提交前逐项检查）

1. `python scripts/validate_finetune.py` 全部通过：
   - 每行可解析为 JSON；
   - 必填字段存在且非空；
   - 关键字段类型正确（form 为对象且键集合正确）；
   - instruction/answer 长度在合法范围（≥5 字，≤1200 字）；
   - 无重复 instruction（去重检查）；
   - form.quantity 等字段不出现空对象。
2. `python scripts/export_finetune.py` 成功导出 `data/dataset.jsonl`，条数与源数据合计一致。
3. 人工抽检：随机 10 条，无编造数字、无承诺报价、口径与知识库一致。
4. 每条数据可追溯到对应场景类别（抽查说明）。

## 6. 扩充到千条级的路径

`scripts/generate_finetune.py` 基于：
- 知识库条目 × 问法模板（≥20 种问法）→ raw_qa 批量；
- 定制场景模板 × 参数组合 → raw_form / raw_dialogue 批量；
- 员工任务模板 × 行业素材 → raw_staff 批量。
生成结果必须经过 `validate_finetune.py` + 人工抽检后方可合入正式集。
