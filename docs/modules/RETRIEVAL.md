# 模块三：语义设定检索

输入事件筛选模块的完整JSON报告，输出逐事件相关设定。此阶段完全不调用LLM，也不判断矛盾或写canon。复用固定的本地BGE小模型与现有48片段索引，不重建、不修改冻结benchmark。

## 内部方法

事件→查询文字→BGE tokenizer→Transformer→CLS pooling→L2归一化→索引向量矩阵点积→稳定降序Top-K。单位向量的点积等于cosine score，没有向量数据库。继承官方查询提示和FP32配置，正文索引与查询编码配置必须匹配。

查询包括主体、事件、叙述性质、明示心理、条件和实际引用的局部原文。引用按原文字符位置排列，保留左右、否定及声称/梦境/计划性质；不把整段故事拼进每条查询。embedding不承诺理解这些差异，只负责候选相关性。

1. 校验输入stage/status、唯一事件ID、全部决定和selected/ignored/pending分区，分区事件必须与原事件相同；错误输入在加载模型前明确拒绝。
2. keep的实质事件独立检索。support_only仅作为required_by_event_ids指向事件的上下文保留，不独立检索；pending/ignored不检索，仍完整留在filtering报告中。三层的原始记录不会被丢弃。
3. 有实际查询才加载BGE和索引；空选择或上游error不加载。缺失/损坏索引先检查，再加载模型，错误报告仍携带完整上游和失败事件。默认项目data/index，只读使用。
4. 默认K=5，超出片段总数返回全部。相同查询在单次请求内缓存结果，保留各自event_id，不合并事实。每条查询单独隔离错误，一条失败不清空其他事件结果。查询含官方提示完整输入须≤512 tokens，不截断，也不把单事实切成独立命题。
5. 每条evidence原样包含id、text、file、start_line/end_line、heading_path及score等索引信息。items还包含query、event、状态和耗时。根报告保留模型/分块配置和资料摘要index、上游filtering、失败原因以及处理状态。

## 状态与限制

items状态是ok、error或support_only。所有实际查询失败或上游error时根status=error；部分查询失败或上游partial时partial；否则ok。retrieval_complete仅为本层查询完成，processing_complete还要求上游完整。空检索结果不证明无矛盾；未检索的pending仍待复核。

semantic_verification固定not_verified。score表示相对相关性，不是正确率或矛盾置信度。没有相关度阈值，完全无关事件也可能返回候选。不对左右、时间或世界规则做关键词纠正，不让LLM改写查询或重排。

## 使用

所有命令在项目根目录执行。完整前三层串联（Qwen只加载一次，检索另用BGE）：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline retrieve --text "雷喝下杯里的水，随即毒发倒地。他的左心脏中，亚巴顿开始躁动。" --device cuda --compact
```

已有完整筛选报告，跳过LLM，只检索：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 -m agent_pipeline retrieve --filter-file .\filter_result.json --device cpu --top-k 5 --compact --output .\retrieval_result.json
```

--file读UTF-8故事，--events-file读第一模块完整报告后筛选/检索，--filter-file读第二模块完整报告直接检索。--index指定已有索引。--retrieval-device可单独指定cpu/cuda，否则随--device。--output不覆盖已有文件；compact缩减终端记录，保存文件仍完整。partial/error退出码2。

API支持复用已加载的Retriever：

```python
from agent_pipeline import retrieve_events
from encoder import Encoder
from retrieval import Retriever

retriever = Retriever('data/index', Encoder(device='cpu'))
report = retrieve_events(filtered_report, retriever=retriever, k=5)
```

## 验证

CPU/GPU观察、索引重载、左右诊断、K超过总数和超长查询的记录见[模块观察报告](../../agent_pipeline/reports/module_retrieval_20260918T083526Z_0f2d54/report.json)。此观察不是正式一致性benchmark，也没有预热后的CPU/GPU速度比较。报告中request_seconds包含本次请求实际发生的工作，不能把首次GPU调用开销用作稳定性能对比。

下一模块judge才将事件及候选证据送给LLM，输出明确吻合/明确矛盾/不确定。

Step 51 通用结构修复与后续边界见[本轮说明](STRUCTURAL_FIXES.md)。
