# 知绣坊 · 刺绣企业本地离线智能客服 Agent

面向刺绣公司的**私有化本地离线 AI Agent**：本地 LLM 推理 + RAG 知识库 + 双角色客服
（C 端客户顾问 / B 端员工助手），全程无公网调用。基于 Qwen2.5 本地模型，
可经 **LlamaFactory LoRA 微调**学习行业话术与需求表单解析；当前挂载开源刺绣行业
公开知识库完成测试，后续可无缝替换为企业私有专属知识库。

> 项目结构参考 `medical_retrieval_plus`（分层工程：数据/NLP/检索/生成/API/前端）。

## 架构总览

```
┌─ 前端  React/Vanilla（双入口：客户咨询页 / 员工工作台）      ← src/embroidery_agent/web
├─ API   FastAPI（角色路由 + SSE 流式 + 工单查询）              ← api/server.py
├─ Agent 角色分流：客户会话（需求追问循环）+ 员工会话 + 工单      ← agent/
│         ├─ 意图分类 → KNOWLEDGE / CUSTOM_REQUIREMENT / ...
│         ├─ 门店专属信息（价格/工期/库存）→ 兜底话术 + 自动工单
│         └─ 定制需求 → 固定顺序追问 → 结构化表单 → 人工确认
├─ 检索   BM25 + FAISS 向量 + bge-reranker 精排 + 权限过滤       ← retrieval/
├─ 生成   Ollama(OpenAI兼容) 流式 + 抽取式兜底 + 提示词模板       ← generation/
└─ 数据   data/knowledge(公开) + data/internal(仅员工) + SQLite 工单
```

## 技术栈与选型（8GB 显存方案）

| 组件 | 选型 | 说明 |
|---|---|---|
| 基座模型 | 任意 Ollama 中文模型（默认 `qwen3.5:9b`，推荐 8G 显存可用 `qwen2.5:7b-instruct`） | 改一行配置即换模型 |
| 微调 | LlamaFactory QLoRA（lora_rank=8, 4bit） | 8GB 显存可训 7B；零代码 Web UI |
| Embedding | bge-m3（Ollama）/ bge-small-zh 起步 | 知识库 <1 万 chunk 用 small |
| Reranker | BAAI/bge-reranker-base（CPU） | 两阶段检索，防检索退化 |
| 向量库 | FAISS（faiss-cpu） | 单机万级 chunk 零运维 |
| 推理部署 | Ollama（OpenAI 兼容 API） | Windows 原生一条命令起服务 |
| 对话/工单 | SQLite | 本地零依赖 |

## 快速开始

### 0. 安装 Ollama 并拉取模型（推理必需）

```powershell
# 1. 到 https://ollama.com 下载安装 Ollama
# 2. 拉取模型（8GB 显存推荐 7B Q4；纯 CPU 用 3B）
ollama pull qwen2.5:7b-instruct
ollama pull bge-m3
```

### 1. 安装后端依赖

```powershell
cd embroidery-agent
pip install -e ".[api,retrieval,dev]"
```

### 2. 构建知识库索引

```powershell
$env:PYTHONPATH = "src"
python scripts/build_kb.py
```

### 3. 启动服务

```powershell
uvicorn embroidery_agent.api.server:app --host 127.0.0.1 --port 8000
```

打开 http://127.0.0.1:8000 （客户咨询 / 员工工作台双入口），
http://127.0.0.1:8000/team 为 **Agent 团队工作台**。

> Ollama 未启动时，系统自动降级为"知识库直接引用"模式（抽取式回答），仍可体验检索链路；
> 团队工具专家（物流/报价/排程）在无 LLM 时直接返回确定性核算结果，业务不中断。

## 多 Agent 团队（B 端生产力工具）

在双角色客服之外，内置 6 个专家 Agent + 团队协调器（`agent/team.py`），
访问 `/team` 工作台或调用 `/api/v1/team/chat`，可自动路由或手动指定专家：

| 专家 | 职责 | 生产可用关键 |
|---|---|---|
| 智能客服助手 | 客户咨询、话术生成 | 团队 RAG 检索增强（bge-m3+FAISS） |
| 物流核算 | 运费试算、承运商比选 | 确定性运费计算器，价格表 `configs/logistics.json` 可直接编辑 |
| 订单报价核算 | 工时成本估算、报价区间 | 成本模型计算器，产能参数 `configs/production.json` |
| 生产排程 | 工序拆解、交期评估 | 排程估算器（打版/绣制/后整/质检） |
| 质检助手 | 疵品归因、返工建议 | 结构化质检报告 |
| 经营数据助手 | 工单统计问答 | 只读 SQL 白名单查询 |

**架构原则**：所有数值计算由纯函数（`agent/business_tools.py`）完成，
LLM 只负责理解需求、提取参数与解释结果，严禁自行算术——模型解释与真实计算天然一致。

**三段式流程**：参数提取 → 确定性工具核算 → LLM 流式解释。
参数提取先走正则（`kg/尺寸/工艺/数量` 直接命中，零 LLM 调用），
缺参才用 LLM JSON 提取且**携带对话历史**——省略句（"那只要 5 幅呢"）能从上文
继承尺寸/工艺/装裱继续核算，不重复追问。参数仍不全时返回中文补参提示，不瞎猜。

**路由**（`agent/team.py`）：关键词正则 → LLM JSON（带 LRU 缓存）→ 客服兜底。
业务句（报价/运费/交期/质检/统计）直接关键词命中，零路由延迟；只有知识问句才走
LLM 路由。日志输出 `路由[keyword|llm|manual]` 便于观测。

**外挂 RAG（已接入向量知识库）**：`agent/team_rag.py` 实现 `agent/rag.py` 的
`RAGProvider` 协议，用 bge-m3 + FAISS 检索 `data/knowledge` + `data/team_kb`，
索引缓存于 `data/index/team_rag`（源文件变更自动重建）。当前 `data/team_kb` 已用
**NCIFD 民族文化数据集**（GitHub 开源，698 条刺绣/工艺问答）填充；换企业私有
知识库时删除该目录放入自有 md/txt，重建索引即可，或改 `configs/default.json` 的
`team.rag_provider` 指向自定义实现。团队客服/质检的回答会自动引用检索内容并标
`[编号]` 来源。RAG 状态可见 `GET /api/v1/team/status`（需口令）。

**提示词热更新**：专家话术在 `prompts/team/*.md`，改完即生效（mtime 感知缓存），
无需重启。文件缺失自动回退代码内置常量。配置同理：`configs/*.json` 改价格/产能
即时生效。

**会话持久化**：团队对话落 SQLite（复用工单库文件），重启不丢，支持跨轮省略句。

**确定性工具直调 API**（不经 LLM，需员工口令 `x-staff-access`）：
`POST /api/v1/tools/logistics`、`POST /api/v1/tools/quotation`、`POST /api/v1/tools/schedule`。

**冒烟测试**（44 项，仅 RAG 检索需本地 Ollama，其余不依赖 LLM）：

```powershell
$env:PYTHONPATH = "src"
python scripts/smoke_team.py
```

**摄取与重建团队知识库**：

```powershell
git clone --depth 1 https://github.com/letsgoLakers/NCIFD %TEMP%\ncifd
python scripts/fetch_team_kb.py --ncifd "%TEMP%\ncifd"
python -m embroidery_agent.agent.team_rag   # 重建向量索引
```

## 双角色权限设计

| 维度 | C 端客户 | B 端员工 |
|---|---|---|
| 知识库 | 仅 public（工艺/针法/品类/保养/纹样寓意） | public + internal（含报价逻辑/工序标准） |
| 能力 | 咨询 + 定制需求登记（工单） | 行业答疑 + 复盘 + 文案 + 工单处理 |
| 话术 | 温柔国风、不承诺、不报价 | 专业术语、可直接使用 |
| 硬约束 | 门店专属信息一律转人工 | 不泄露内部成本/底价 |

## 关键流程

**定制需求追问循环**（客户意图识别为 CUSTOM_REQUIREMENT 时）：
尺寸 → 工艺类型 → 使用场景 → 预算 → 交付时限 → 特殊要求（每轮仅问 1 个）
→ 信息齐全输出【结构化需求表单】+【方案建议】+【待人工确认】+ 自动建工单。

**客户确认回环（human-in-the-loop）**：六要素齐 → 表单先给客户过目 → 客户回复「确认」才转人工建单，回复修改意见则更新表单再次确认（需求抽取采用「规则先行、LLM 补空」策略，高频表达零模型命中）。

**兜底规则**：客户询问价格/工期/库存等门店专属信息 → 固定话术转人工 + 自动工单。

## 微调（LlamaFactory）

1. 整理行业对话数据至 `data/finetune/`（raw_qa / raw_dialogue / raw_staff / raw_form）
2. 导出 LlamaFactory 训练集：`python scripts/export_finetune.py`
3. LlamaFactory QLoRA 微调 Qwen2.5-7B → 导出 GGUF → `ollama create` 导入
4. 修改 `configs/default.json` 的 `chat_model` 指向微调模型

## 评测

```powershell
python scripts/evaluate.py    # 三合一：检索 Hit@K/MRR + 意图准确率 + 行为合规
```

黄金测试集 `data/golden_test.jsonl` 每行标注 {query, expected_sources, role, intent, behavior}，
覆盖知识咨询、纹样寓意、定制需求、价格转人工、投诉、员工复盘/文案等全部意图。

## 微调数据（v2）

规格文档：`docs/finetune_data_spec.md`（字段格式/数量目标/质量红线/验收标准）

```powershell
python scripts/validate_finetune.py            # 四类原始数据格式校验
python scripts/validate_finetune.py --export   # 校验通过后导出 data/dataset.jsonl
python scripts/generate_finetune.py --per 2    # 问法变体扩充（输出到 generated/，人工抽检后合入）
```

## 质量保障

```powershell
python -m pytest tests -q      # 单元测试（意图/工单/会话落库/BM25，全部离线可跑）
python scripts/smoke_team.py   # 端到端冒烟（含鉴权回归，需本地 Ollama）
python scripts/evaluate.py     # 综合评测（需本地 Ollama 与 bge-m3）
```

CI（GitHub Actions）已配置于 `.github/workflows/ci.yml`：
单元测试 + 数据校验 + 问法扩充自检 + 模块导入检查，全程不依赖 Ollama。

## 部署（客户机 Windows）

显存方案（按开发机 12G 为基准、客户机 16G 留余量）：
qwen2.5:7b-instruct(Q4 约5.5G) + bge-m3(约1.2G) + bge-reranker-base(约1G) ≈ 7.7G，开发机余量充足。

一键安装/升级（管理员运行，检查环境→装依赖→建索引→注册开机自启→注册每日备份）：
```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```
- 服务自启：NSSM 注册 Windows 服务（nssm.exe 需与脚本同目录或加入 PATH）；
- 每日备份：schtasks 注册 ZhixiufangBackup（每天 03:00 执行 backup.ps1，保留最近 14 份）；
- 手动备份：`powershell -File backup.ps1`；
- 前台启动（无服务注册时）：`.\start.ps1`。

## 部署（备选 Docker）
  ```powershell
  docker build -t zhixiufang .
  docker run --rm -p 8000:8000 --add-host=host.docker.internal:host-gateway `
    -e EMBROIDERY_LLM_BASE_URL=http://host.docker.internal:11434/v1 `
    -e EMBROIDERY_EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1 `
    -e EMBROIDERY_STAFF_PASSCODE=请替换为强口令 zhixiufang
  ```

## 渠道接入（微信/直播等多端）

代码层已完成"渠道适配层"，把知绣坊客服接到外部平台：

- `src/embroidery_agent/channels/`：渠道抽象（ChannelAdapter）+ 微信适配器 + openid 绑定 + 消息幂等 + 异步回复队列；
- webhook 端点：`GET/POST /api/v1/channels/{channel}/webhook`（GET 验签返回 echostr，POST 收消息立即回空串、后台异步应答）；
- 回复走"客服消息"异步推送；未配置平台凭证时进入本地模式（回复存 `channel_pending_replies` 表，供联调轮询）；
- openid↔会话绑定在 SQLite（同库），同一用户在平台侧始终同一会话，跨设备不丢上下文；
- 微信 5 秒被动回复硬限制已规避（异步应答，实测 0.03s 返回）。

配置（`configs/default.json` → channels）：
- `token`：与公众号后台「服务器配置」一致；
- `verify_signature`：本地联调 false，生产必须 true；
- `app_id` / `app_secret`：填写后启用微信客服消息推送，留空为本地模式。

验证：
```powershell
python -m pytest tests -q                 # 含渠道单测（XML/验签/绑定/幂等）
python scripts/smoke_channels.py          # 端到端：模拟微信服务器回调全链路
```

## 安全配置（生产必读）

- 员工口令三种注入方式（安全性递增）：配置明文 `staff.passcode`（开发）→ 配置哈希 `staff.passcode_sha256`（用 `python scripts/hash_passcode.py 口令` 生成）→ 环境变量 `EMBROIDERY_STAFF_PASSCODE`（最高优先级）；
- 所有员工接口（status/tickets/search(staff)/staff chat/team chat/kb build/metrics/tools）统一鉴权（hmac 常量时间比较）；
- **速率限制**：按 IP 滑动窗口限流（默认对话 10 次/分、检索 30 次/分），配置见 `rate_limit` 段，超限返回 429；
- **提示词注入防御**：`nlp/guard.py` 规则拦截"忽略指令/越狱/套取系统提示/套取底价"类话术并记录审计日志，配合 system prompt 安全防线双层防护；
- 工单前端使用 DOM 渲染，杜绝存储型 XSS；
- 依赖版本已锁定于 `requirements-lock.txt`（Windows 开发机快照，容器内用 pyproject 声明安装）；
- **降级可观测**：模型缺失/生成失败/向量检索失败均写入 `logs/app.log`（WARNING 级），`/api/v1/status` 的 `llm_ready` 反映真实模型可用性（会检测模型是否已安装）。

## 多功能 Agent（已落地）

- 客服问答 / 定制需求收集 / 工单：核心闭环；
- 报价 / 工期估算：确定性工具，客服可直接用；
- 客户意向分级（lead_score）：工单自动打"高/中/低"意向标签 + 原因，
  帮客服排跟进优先级（规则可审计，见 `src/embroidery_agent/agent/leads.py`）。

## 监控

- 日志落盘：`logs/app.log`（服务启动后自动创建，同时保留控制台输出）；
- 进程指标：`GET /api/v1/metrics`（需员工口令）返回运行时长/请求数/错误数。

## 目录结构

```
embroidery_agent/
├── configs/default.json          # 运行配置（模型/知识库/检索/限流/团队）
├── data/
│   ├── knowledge/                # 公开知识库（25 篇：工艺/针法/品类/保养/纹样/场景…）
│   ├── internal/                 # 内部知识库（仅员工可见：报价逻辑/客服SOP）
│   ├── team_kb/                  # 团队外挂知识库（NCIFD 数据集 + 仓库整理，见 SOURCES.md）
│   ├── finetune/                 # 微调原始数据（qa/dialogue/staff/form）
│   ├── golden_test.jsonl         # 黄金测试集（146 条，评测用）
│   └── dataset.jsonl             # 导出的 LlamaFactory 训练集（生成物）
├── scripts/
│   ├── build_kb.py               # 构建知识库索引
│   ├── evaluate.py               # 综合评测（检索+意图+行为合规）
│   ├── validate_finetune.py      # 微调数据校验
│   ├── generate_finetune.py      # 问法变体扩充
│   ├── export_conversations.py   # 数据飞轮：真实对话→微调候选
│   ├── hash_passcode.py          # 生成口令 SHA-256
│   ├── smoke_team.py             # 端到端冒烟测试（含鉴权回归）
│   └── export_finetune.py        # 导出微调数据
├── tests/                        # pytest 单元测试（46 项，离线可跑）
├── .github/workflows/ci.yml      # CI 流水线
├── Dockerfile                    # 容器化部署
├── requirements-lock.txt         # 依赖版本锁定（Windows 开发机快照）
├── prompts/team/                 # 专家提示词（热加载）
├── src/embroidery_agent/
│   ├── core/                     # schema + settings
│   ├── nlp/                      # 工艺实体识别 + 意图分类 + 注入防御（guard）
│   ├── retrieval/                # BM25 + FAISS + reranker + pipeline（含源文件变更检测）
│   ├── agent/                    # customer / staff / ticket / router / team / experts / leads / business_tools
│   ├── channels/                 # 渠道接入层（微信 webhook / openid 绑定 / 幂等）
│   ├── generation/               # prompts + llm + answer
│   ├── api/                      # FastAPI + SSE + 统一鉴权 + 限流 + 监控指标 + webhook
│   └── web/                      # 前端页面（客户页 + 团队工作台）
└── pyproject.toml
```

## 免责声明

当前为开源知识库测试阶段：所有应答仅作行业参考，不包含门店专属权益、价格、工期承诺；
不替代人工专业判断，敏感业务决策一律人工兜底。
