# 团队知识库数据来源声明

本目录（`data/team_kb/`）是团队多 Agent 的**外挂知识库**挂载点，通过
`configs/default.json → team.rag_kb_dirs` 挂载，由 `team_rag.VectorRAGProvider`
独立建索引（`data/index/team_rag`）。替换为企业私有知识库时，删除或替换本目录
文件后重建索引即可，代码零改动。

## 当前数据源

### 1. NCIFD 民族文化数据集（外部开源）
- 仓库：https://github.com/letsgoLakers/NCIFD
- 文件：`ncifd_刺绣工艺.md`（面向大模型的民族文化指令数据集，按刺绣/工艺/纺织关键词离线过滤）
- 许可：以原仓库声明为准，仅用于学习研究

### 2. 本仓库整理编写（原创）
- 文件：`纹样民俗百科.md`、`行业经营常识.md`
- 说明：综合公开民俗资料与行业通用知识整理，不含真实门店数据，可自由使用

## 扩充指引

1. 新文件放入本目录（.md/.txt），文件名即分类主题；
2. 重建索引：`python -m embroidery_agent.agent.team_rag`；
3. 更新本声明（来源 + 许可），保持溯源习惯。
