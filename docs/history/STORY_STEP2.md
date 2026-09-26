# Story-Level Consistency Benchmark：逐事实检索

这一小步接通：故事 → 重要事实建议 → 每条事实的设定 Top-K。
尚未判断一致/矛盾/不确定，未保存世界观，也不是正式 benchmark 成绩。

## 体验命令

```powershell
Set-Location "C:\Users\Xhang\Desktop\project\worldcheck"
& ".\.venv\Scripts\python.exe" -X utf8 ".\story.py" retrieve --text "雷捂住左胸，感到寄宿其中的亚巴顿开始躁动。" --device cuda
```

没有 GPU 可选 `--device cpu`；`auto` 自动选择。CPU 上 Qwen 提取会慢。
文件输入用 `--file "故事.txt"` 替代 `--text`，支持 UTF-8/BOM。
`--top-k 5` 默认每事实返回五条；`--index` 可指定已有索引；`--json` 输出完整诊断。
原有 `extract` 命令保持可用。

## 内部结构

`story_retrieval.py` 独立于冻结的 Fact-Level 实现，直接复用已有 Encoder 和 Retriever。
`retrieve_story(...)` 先调用提取器，只有有合法保留事实时才加载 BGE 和索引；模型、索引各复用一次。
`retrieve_facts(extraction, retriever, k=5)` 接收提取报告，方便后续流水线复用及独立测试。

每条查询由该事实的 subject/predicate/object、source_text、选定 context_text 和 time 组成。
不放整段故事，不放其他候选，不用猜测的新实体或扩写。局部必要上下文可能包含另一个实体，用于保留明确指代和条件。
BGE 沿用 FP32、官方查询提示、CLS pooling、L2 归一化。每条查询编码后独立做 NumPy 点积排序；没有新增 LLM 检索调用。
不截断超过编码上限的查询，保留明确错误；该事实失败不会阻止其他事实检索。

只有 extraction.facts 参与检索，discarded/rejected/pending_facts 不参与。
原始提取报告完整嵌入 extraction；items 中每条包含 fact、query、evidence、status、retrieval_seconds，以及失败时 error。
证据保持原始片段 ID、正文、文件、起止行、标题路径和 cosine score。
无重要事实合法返回空；提取 partial 不因检索成功被改为 ok；检索失败也不伪装成成功空结果。

相似度只是相关性，不是事实正确率或矛盾置信度。完全无关内容仍可能有 Top-K。

## 本地开发检查

真实本地 CUDA CLI 完整试跑报告：reports/story_step2_retrieval_smoke.json。
输入“雷捂住左胸，感到寄宿其中的亚巴顿开始躁动。”，提取两条、分别返回五条证据。
F2 的 Top-1 为“亚巴顿寄宿在雷的左侧心脏里……”（cosine 0.6695），Top-2 为双心脏及两位天使宿主设定（0.6349）。
F1 “捂住左胸”仍被提取器保留，属于尚未解决的重要性误提；其检索首位也可能是右胸治疗反应，不能据此判断一致。

新增五项测试覆盖查询隔离、证据绑定、空提取、单条失败隔离、条件/null 对象及选项错误。
全套 90 项回归测试通过，38 个冻结 Fact-Level 文件 SHA256 不变。
下一小步才接逐事实一致性判断：仅提交当前事实、必要条件及它自己的证据，不提交整段故事。
