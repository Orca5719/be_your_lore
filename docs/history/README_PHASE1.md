# WorldCheck

第一个实验：测试资料不代表正式 canon。

在项目目录中运行：

```powershell
uv sync --locked
.\.venv\Scripts\python.exe -X utf8 experiments\01_sentence_vectors.py
```

首次运行下载 BGE 模型到项目 .cache/huggingface；只下载公开模型，不上传示例文本。
缓存完整后可在无网络下运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments\01_sentence_vectors.py --offline
```

用 --device cpu 或 --device cuda 指定设备。默认自动选择。
实验展示 token ID、补齐位置、模型输出形状、CLS pooling、单位向量和余弦相似度。
这里是句子相似度实验，三个输入都按正文编码；之后检索才为查询添加官方检索提示。
reports/experiment01.json 保存实测配置与结果，experiment01_vectors.npy 保存三条向量。
依赖由 pyproject.toml 和 uv.lock 固定。GPU 版 PyTorch 来自官方 cu126 源，无需单独安装完整 CUDA Toolkit。
目前已实现分块、建库、检索和统一命令入口；示例库已扩展至四类、48 个片段，20 条固定查询评测已完成。

## 实验 02：文档读取与分块

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments\02_document_chunks.py
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

--lore 指定资料目录，--max-tokens 指定 3 至 512 的输入预算（默认 512）。
当前按 # 到 ###### 标题与空行分段，递归支持 .md/.txt，包含 UTF-8 BOM。
标题上下文加入 embedding_text，text 保持正文；输入长度包含标题和特殊 token。
长段落按句子边界组合；最后一句及下一句能共同放入时保留一句重叠。
超长单句按字符边界切开，仍用 tokenizer 验证；预算不足时不重叠，保证继续推进且不丢正文。
来源偏移基于读取后的逻辑 Unicode 文本（BOM 移除，CRLF 归一为 LF），不是字节偏移。
reports/chunks.json 保存片段原文、标题、行号、偏移、编码输入和 token 数。
当前示例库已有 48 个片段并生成文档向量，实验 02 可查看全部分块。

## 实验 03：批量编码与索引保存

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments\03_build_index.py
```

实验离线加载固定 revision 的 BGE 模型，默认自动选择 CUDA、FP32、batch_size=16。
encoder.py 是后续建库/检索共用的编码器；正文不加查询提示，encode_queries 为查询添加官方提示。
index_store.build_index(lore_directory, index_directory, encoder, max_tokens=512, batch_size=16) 建库。
index_store.load_index(index_directory, expected_config=None) 返回向量矩阵与元数据。

索引结构：

```text
data/index/
  CURRENT                       当前版本 ID
  versions/<版本 ID>/
    embeddings.npy              [片段数, 向量维度] FP32
    metadata.json               同序片段、模型/分块配置、资料摘要
    manifest.json               两个文件的 SHA-256 摘要
```

读取器核对摘要、片段数量、唯一 ID、模型维度、有限数值与单位长度。
新版本完整写入并验证后，以 os.replace 切换 CURRENT；失败不会改变旧版本指针。
旧版本暂时保留，不自动清理。该发布方式避免两文件部分更新；尚不承诺断电恢复。
源文件摘要用于记录资料版本；建库期间资料变化会拒绝发布。资料更新后仍须显式重建。
实验会再次编码并核对磁盘重载结果；这次重复编码仅用于教学验证，正常建库无需重复。
查询和统一命令入口见下方说明。

## 实验 04：查询与 Top-K 检索

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments\04_search_lore.py
.\.venv\Scripts\python.exe -X utf8 experiments\04_search_lore.py "雷的左心脏住着谁？" --k 2
.\.venv\Scripts\python.exe -X utf8 experiments\04_search_lore.py "雷的左心脏住着谁？" --json
```

retrieval.search(query, index_directory, encoder, k=5, max_tokens=512) 返回结构化片段列表。
重复查询建议构造 Retriever(index_directory, encoder)，只加载一次模型与索引，再调用 .search(query)。
查询使用官方检索提示，设定向量不重新编码。归一化矩阵 E @ q 得到每个片段的 cosine score。
采用稳定降序排序，相同分数保持原片段顺序；K 超过片段数量时返回全部。
结果包含 id、text、file、start_line、end_line、heading_path 和 score，同时保留原索引字段。
空查询、非法 K、超长查询，以及模型或分块配置不匹配会被拒绝。
当前 max_tokens 默认 512；若未来以其他预算建库，检索必须传入同样的预算。
reports/experiment04.json 保存最近一次查询，experiment04_checks.json 保存真实模型检查结果。
可手动运行 tests/check_real_retrieval.py 验证真实模型，不放入快速单元测试默认流程。
当前不设相关性阈值，无关查询仍返回 Top-K；输出附带相关性声明。
当前已有 48 片段/20 查询评测；统一 CLI 和交互会话见下方说明。

## 统一命令入口

在项目目录运行（默认路径始终指向项目 lore 和 data/index，不受终端当前目录影响）：

```powershell
# 资料更新后显式重建索引
.\.venv\Scripts\python.exe -X utf8 worldcheck.py index

# 单次查询
.\.venv\Scripts\python.exe -X utf8 worldcheck.py search "雷感觉左胸里的亚巴顿开始躁动。" --k 2

# 连续查询，输入 /quit 或 /exit 退出
.\.venv\Scripts\python.exe -X utf8 worldcheck.py interactive

# JSON 输出
.\.venv\Scripts\python.exe -X utf8 worldcheck.py search "德尔塔装甲靠什么供电？" --json

# 查看命令参数
.\.venv\Scripts\python.exe -X utf8 worldcheck.py --help
```

共同选项：--index、--device auto/cpu/cuda、--max-tokens（须与建库一致）、--json。
默认只读取本地模型缓存；确实需要下载公开模型时添加 --download。
index 另外接受 --lore 与 --batch-size（默认 16）；search 和 interactive 接受 --k（默认 5）。
连续模式只初始化一次 Encoder 和 Retriever，循环内仅查询；空输入报错后继续，EOF 或 Ctrl+C 可退出。
连续 JSON 模式为 JSON Lines（每个成功查询输出一行 JSON），错误与终端提示在 stderr，不混入 stdout。
默认文字模式展示原文、标题、文件行号和分数；JSON 返回 query、notice 和 results。
参数错误和加载/查询失败退出码为 2；正常运行/退出为 0。
已有 experiments/ 教学入口保留，方便观察中间形状。

真实命令验证：

```powershell
.\.venv\Scripts\python.exe -X utf8 tests\check_real_cli.py
```

它使用临时索引验证建库、JSON、连续查询、空行恢复与退出，不替换正式 data/index；报告在 reports/cli_checks.json。
inspect 和 benchmark 已实现；后续应在真实作品资料上进一步评测。

## 固定查询集评测

示例库为人物、天使、历史、科技四类，共 48 个片段；全部是测试设定，不代表正式 canon。
evaluation/queries.json 在检索前固定 20 条查询，每条标注唯一核心证据与零到多条补充证据。
标签使用文件和标题定位；评测前要求每个标签唯一匹配一个片段。

```powershell
# 现有索引与资料一致时直接评测
.\.venv\Scripts\python.exe -X utf8 evaluate.py

# 修改测试资料后，重建正式索引并评测
.\.venv\Scripts\python.exe -X utf8 evaluate.py --rebuild
```

程序检查资料摘要，发现索引过期会要求重建，不把过期索引结果当作当前评测。
报告在 reports/retrieval_evaluation.md 和 reports/retrieval_evaluation.json。
指标：核心证据 Top-5 命中率、完整找齐证据的查询数、全部标注证据的找回率。
阶段目标为至少 16/20 核心命中，且指定左胸亚巴顿查询命中；退出码 0 表示达到目标，1 表示未达到。
首次实测：核心 20/20，全部证据齐全 19/20，总证据 30/31（96.8%）；q20 漏掉莉娅工作职责补充证据。
q01 双心脏结构核心证据排第 5，仅能说明当前 Top-5 找回，不能说排第一或永远可靠。
本次未根据模型结果修改资料或标签。小型合成库的结果不能代替真实作品上的泛化评测。
evaluation/prepare_sample.py 是一次性测试夹具生成脚本，会覆盖四个示例文件；日常使用不需要运行，接入真实资料后不要运行它。


## inspect：查看索引和编码过程

```powershell
# 不加载模型，只检查磁盘索引
.\.venv\Scripts\python.exe -X utf8 worldcheck.py inspect

# 查看任意正文的 token、形状、CLS 与归一化
.\.venv\Scripts\python.exe -X utf8 worldcheck.py inspect "雷的左心脏寄宿着亚巴顿。"

# 查询模式：观察检索提示实际增加了哪些 token
.\.venv\Scripts\python.exe -X utf8 worldcheck.py inspect "雷的左心脏寄宿着亚巴顿。" --mode query

# 结构化输出
.\.venv\Scripts\python.exe -X utf8 worldcheck.py inspect "雷的左心脏寄宿着亚巴顿。" --mode query --json
```

带文本时默认 passage 模式；query 模式使用与正式检索相同的提示。
检查完整编码输入，超过 512 tokens 会拒绝，不截断；索引查看不加载模型、不重新建库。
展示向量前 8 维及长度；单个维度不能直接解释为角色年龄、心脏数量等属性。
inspect 与正式编码共用 _forward 路径，避免教学输出偏离实际工具。
输入一条文本时 B=1，通常 attention_mask 全为 1；不同长度批量文本的补齐可看实验 01。
真实检查脚本 tests/check_real_inspect.py，实测报告 reports/inspect_checks.json。

## 后续录入方式

用户资料将由用户录入，内部按实体、类别和条目组织；当前文件式 lore 是实验输入。
要求记录在 PRODUCT_NOTES.md，本阶段暂不实现录入/分类界面或修改数据模型。


## benchmark：性能与精度比较

```powershell
.\.venv\Scripts\python.exe -X utf8 worldcheck.py benchmark
# 增加重复次数；日志在 stderr，JSON 结果在 stdout
.\.venv\Scripts\python.exe -X utf8 worldcheck.py benchmark --repeats 5 --json
```

默认同一索引的全部片段，CPU FP32 / GPU FP32 / GPU FP16 各比较 batch=1 和 batch=16。
每组预热完整资料一轮，测量三轮取中位数；--warmups、--repeats、--threads 可调整。
默认 CPU intra-op 线程为 4，报告记录原线程数，运行结束恢复；无 CUDA 时明确跳过 GPU 组。
每个 device/precision 加载一次模型，两个 batch 共用同一加载时间。模型加载与质量检查单独计，不混入编码耗时。
FP16 仅用于实验模型推理，L2 归一化和返回 NumPy 向量仍为 FP32；正式索引和普通命令保持 FP32。
benchmark 对每组都重新编码资料与 20 查询，比较向量误差、Top-5 顺序/集合及标注证据找回。
当前基准使用固定示例查询标签，所以 --index 必须能匹配这些标签，不能用任意真实库直接替代。
报告在 reports/benchmark.md 与 reports/benchmark.json，包含每轮原始样本、加载时间和硬件/模型配置。

GPU 计时前后使用 [torch.cuda.synchronize](https://docs.pytorch.org/docs/2.7/generated/torch.cuda.synchronize.html) 等待任务完成。
每轮重置峰值统计，记录 [max_memory_allocated](https://docs.pytorch.org/docs/2.7/generated/torch.cuda.max_memory_allocated.html) 及保留空间峰值。
这些是当前进程的 PyTorch tensor/allocator 显存统计，不能等同于整张显卡总占用。

首次 RTX 4060 Laptop 实测（48 短片段、CPU 4 线程、三轮中位数）：

| 配置 | batch | 全批耗时(ms) | tensor峰值(MiB) |
|---|---:|---:|---:|
| CPU FP32 | 1 | 711.3 | — |
| CPU FP32 | 16 | 492.9 | — |
| GPU FP32 | 1 | 220.3 | 102.5 |
| GPU FP32 | 16 | 34.8 | 128.2 |
| GPU FP16 | 1 | 216.9 | 55.5 |
| GPU FP16 | 16 | 27.6 | 69.1 |

GPU FP32 批处理相对逐条编码约快 6.33 倍；GPU 批处理 FP16 相对 FP32 约快 1.26 倍。
全部配置核心 20/20，完整证据 19/20。FP16 batch=16 的 q19 Top-5 顺序变化，但集合相同，核心仍排第一。
这些数值只描述本次短文本实验；固定测试顺序，未控制笔记本温度、功耗或其他负载，不能承诺同样倍数用于长文本/单次查询。
首个配置是 CPU FP32 batch=1，全部质量比较以它为基线。
最重要的观察是 batch=16 将 48 次模型调用减少为 3 次；GPU 单条 FP16 几乎没提速，不能以省显存推断必然大幅加速。

### 第二阶段学习实验：Qwen 本地生成

首次运行（下载公开模型权重）：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\experiments\05_qwen_generation.py --download --device cuda
```

缓存齐全后去掉 `--download` 即离线运行。可选 `--device auto/cpu/cuda`；GPU 使用 NF4 双重量化和 BF16 计算，CPU 使用未量化 BF16，CPU 路径尚未实测。`--prompt` 可更换简短问题。输出 `reports/experiment05_qwen.json` 展示聊天模板、token IDs、attention mask、形状及实际回答。单次生成耗时不是预热后 benchmark；本实验尚不判断世界观一致性。

### 手工证据一致性实验

```powershell
.\.venv\Scripts\python.exe -X utf8 .\experiments\06_manual_judge.py
```

本实验只用本地缓存与 CUDA，模型一次加载后检查三个固定开发例子（支持、左右颠倒、新能力）。实验 05 保留 CPU/auto/cuda；完整判断模块的 CPU 选择将在接入时沿用。报告保存在 `reports/experiment06_manual_judge.json`，含完整提示、证据、原始输出、校验结果及单次耗时。

`judgment_protocol.py` 检查逐条判断、输入原文、引用 ID 与逐字证据，然后按“矛盾 > 不确定 > 一致”汇总。无效 JSON、假引用和生成未结束均记录为独立 error，不自动重试、不当作不确定。引用合法不能保证推理正确；新增候选仅针对提供的证据，不自动写入资料。这三个样例不是正式 baseline。

### 作者录入：分类预览、判断与确认保存

在项目目录运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\entry.py "雷能控制时间。" --entity 雷 --category 能力 --device cuda
```

先显示已有同类条目、其他类别、拟新增原文、判断与证据来源。输入 **确认保存** 才写入，其他输入或 EOF 均取消。只想体验判断而不保存，添加 `--preview-only`。`--json` 输出分行 JSON，确认提示在 stderr。

经历或时间线可附事件时间，例如：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\entry.py "雷在大地震后失忆。" --entity 雷 --category 经历 --time "大地震后" --device cuda --preview-only
```

时间会同时送入检索与判断；未指定时间不自动推断。默认由本地 Qwen 识别实体、建议类别并复制原文明示的事件时间；手动 --entity/--category/--time 可覆盖建议。尚未实现别名归并或时间线排序。已有资料按末两级标题的实体/类别精确读取，新录入按结构化字段读取；不会把双心脏自动并入能力，也不会虚构已有能力。

持久化源是 `data/entries.json`，每条含 ID、实体、类别、原文、事件时间、UTC 录入时间。生成的 `lore/_author_entries.md` 只是检索适配文件，请编辑原始条目而非生成文件。确认后先原子保存 JSON，再导出文本与重建索引；这三步不是跨文件事务。同步失败时明确报告“保存成功、索引同步失败”，不要重复录入，可恢复同步：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\sync_entries.py
```

模型检查失败不允许本次保存；不确定或矛盾都会显示原因，由作者明确确认是否建立这条设定。追加不自动删除或覆盖冲突旧条目。作者确认成功后才显示“新设定已记录”。重复作者条目拒绝；预览后资料改变要求重新预览。Qwen 默认离线，本地 snapshot 路径避免辅助联网；保留 auto/cpu/cuda，CPU 为未量化 BF16，CPU 判断尚未实测。检索及重建使用 CPU FP32，GPU 留给 Qwen。

隔离集成实验：`tests/check_real_entry.py`，报告 `reports/entry_checks.json`。开发验证不等于正式 baseline。


### 直接输入设定：自动识别与分类

现在可省略人物与类别参数：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\entry.py "雷能控制时间。" --device cuda
```

本地 Qwen 先建议人物/实体、类别和原文中明确出现的事件时间，再复用同一个模型判断一致性。预览显示建议，确认人物、类别和时间都正确后输入“确认保存”。建议不对可取消后用 --entity/--category/--time 修正。显式指定人物和类别时保留原有手动模式。

“雷拥有两颗心脏”归生理结构；“第三话，雷失去了长剑”归雷的经历，事件时间第三话。“他能控制时间”主体不明确，终端会请求补充；只预览或非交互输入时返回 needs_clarification 并要求显式字段，不保存代词人物。新人物名可识别，但必须来自原输入。自动类别从通用类别与已有类别选择，作者可手动指定自定义类别。当前一次录入一个归档主体，多人物或多类别请拆开录入。

分类 JSON 与输入来源由 classification_protocol.py 校验，代词由程序确定转为待澄清。分类只是建议，不代表真伪、已保存或高置信度。报告 reports/classification_checks.json 是五个开发案例的真实原始输出加程序校验结果，模型原始实体识别四个正确、代词一个需程序修正，不是正式 baseline。

### 连续录入：模型只加载一次

```powershell
.\.venv\Scripts\python.exe -X utf8 .\entry.py --interactive --device cuda --preview-only
```

启动后直接逐行输入新设定，`/quit` 退出。去掉 `--preview-only` 可逐条预览并确认保存。Qwen 和 BGE 在进程内复用，索引每条重新读取以看见刚保存的条目；退出程序后模型释放，再启动仍需加载。`--json` 输出逐条 JSON。

检查失败现在显示具体原因并标明未保存，不能当作不确定或矛盾。给模型的证据用 E1/E2 短编号，程序严格映射回真实片段 ID，再校验逐字引用，不采用近似 ID 修补。

一致性提示升为 consistency-v4：要求必要的少量证据与简短原因，特别核对频率、程度、时间、条件等限定信息。“出现过疼痛”不能证明“周期性疼痛”。分类与检查分别记录生成耗时/tokens，首次模型加载单独记录；单次计时不是正式 benchmark。

真实连续回归报告 `reports/continuous_entry_checks.json`，开发样例不代表整体准确率。

### 通用多条录入：实体、关系、规则、事件和时间

```powershell
.\.venv\Scripts\python.exe -X utf8 .\entry.py --world "第九话，雷加入晨星议会，并获得控制时间的能力。" --device cuda --preview-only
```

可提议组织实体、时间节点、隶属关系、能力变化等多个关联条目。支持六种基本记录：实体、属性、关系、规则、事件、时间节点；实体类型与类别允许自定义，例如地点、种族、技术、信仰、语言、制度、资源、物理规则等。当前这是通用模式入口，旧单条模式仍保留。

连续体验：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\entry.py --world --interactive --device cuda --preview-only
```

去掉 `--preview-only` 才会询问保存。“确认保存”选择全部；“确认保存 1,3”只保存指定建议；其他输入取消。一次最多 20 条；复杂长输入请分段，现阶段仍受中文短查询的 512 token 限制。LLM 可能漏提取或误分类，预览需审核，不能保证覆盖一切创作内容。

每条保存 kind、主体、建议实体类型、类别、输入原文片段、关系对象、事件时间、有效起点/终点，以及 ID、UTC 创建时间和完整原输入。时间字段只取原文明示的标签，不补日期；尚未实现日历/章节自动排序、区间计算和别名消歧。时间范围会作为证据文字参与判断，但不是确定性的时间推理引擎。

新文件 `data/world_records.json` 是通用条目源。旧 `data/entries.json` 和旧 Markdown 保留，完整目录合并读取；新实体/时间标签查全量目录的精确提及，不将 Top-5 漏检当成实体不存在。生成 `lore/_world_records.md` 供现有检索使用，确认后一次批量原子保存，再导出/重建；跨 JSON、文本与索引不是事务，同步失败明确报告已保存，可用 `entry.py --world --sync-only` 恢复。

每条事实独立调用同一个本地模型判断，避免把整段总判断套到所有条目；耗时随事实数增加。模型只生成 verdict、reason 和 E 编号，程序从真实片段绑定原文与来源，公共接口仍返回完整引用。实体/节点的结构建议单独标为待作者确认，不靠模型直接断言全库真假。未被检查覆盖或检查错误的条目不能保存，可选择其他成功条目。

修正建议可将 `{ "records": [...] }` 保存为 JSON，保留原输入，再通过 `entry.py --world "原输入" --plan "计划.json"` 重新预览和检查。名称、时间、原文片段仍须可在输入定位；类型/类别可调整。此入口是文件方式，尚无逐字段编辑界面。

开发验证报告 `reports/general_entry_checks.json`，使用隔离测试资料；不是正式 baseline。正式 baseline 暂停，待通用结构和时间处理完成后扩展标注覆盖面。

通用提取现采用 `world-extraction-v4`：程序给按标点切分的原文片段编号，模型选择编号、提出类型和主体，程序取回原文。省略主体的“并结识克拉克”不会被补写成“茱莉亚结识克拉克”。短字段只用于模型内部输出；保存与人工 `--plan` 仍使用完整字段。

本机开发复测：茱莉亚例句冷启动约 32.6 秒，同进程复用约 11.4 秒；地点+规则例句复用约 7.4 秒。报告 `reports/extraction_v4_checks.json`，仅作开发验证，耗时随输入、建议数量和设备变化。控制台显示归档和每条事实的耗时。首次加载仍有开销，建议多次体验使用连续模式：

```powershell
& "C:\Users\Xhang\Desktop\project\worldcheck\.venv\Scripts\python.exe" -X utf8 "C:\Users\Xhang\Desktop\project\worldcheck\entry.py" --world --interactive --device cuda --preview-only
```

出现多个事实共享同一片段、复杂标点或识别遗漏时，需拆分输入或编辑 `--plan`；不会自动改写原文。预览仍需审核分类、遗漏和时间范围，不能把不确定当成已建立的新 canon。

## 通用录入 baseline v1

```powershell
.\.venv\Scripts\python.exe -X utf8 .\llm_baseline.py --device cuda
.\.venv\Scripts\python.exe -X utf8 .\llm_judge_baseline.py --device cuda
```

第一条测真实通用录入端到端预览；第二条用人工正确拆分的相同核心事实诊断检索与判断，不能拿第二条成绩替代应用整体成绩。均使用临时示例副本、独立索引与空存储，不保存设定。执行会覆盖对应版本的报告，报告包含当次数据/提示/实现内容摘要及原始输出，比较版本前应保留旧报告。

数据：`evaluation/llm_baseline_v1.json`（28条输入、34条核心事实；26条新组合+2条既有开发回归），诊断标注：`evaluation/llm_judge_baseline_v1.json`。均由实现者运行前人工标注，没有独立专家复核；使用开发过程中已有的示例世界观，不能称真实创作场景或严格独立盲测。

结果：`reports/llm_baseline_v1.md` / `.json` 和 `reports/llm_judge_baseline_v1.md` / `.json`。端到端22/34事实判定正确，18/28有效预览，16/28全项满足，失败不自动重试。速度表中的全部后续输入包含失败请求，阅读时须同时看成功请求的耗时，不能把快速报错当成性能提升。

引用ID及原文有效不保证推理原因正确。核心证据按人工指定原文子串计分，类别只评允许的kind；尚未完整评估实体类型、别名、复杂时间区间、长文本或指令攻击。CPU LLM性能本轮没有测。

两轮最终结论见 `reports/llm_baseline_summary_v1.md`：人工拆分诊断31/34正确，核心证据18/18进入Top5；端到端依然22/34。成功请求复用中位约5.8秒，单条人工拆分事实检索+判断约1.7秒，二者不是同一计时单位。本轮保留已定位错误，没有据此调整提示或标注。

## 160条世界观 benchmark

```powershell
.\.venv\Scripts\python.exe -X utf8 .\world_benchmark.py --validate-only
.\.venv\Scripts\python.exe -X utf8 .\world_benchmark.py --device cuda
```

冻结数据：`evaluation/world_benchmark_160_v1.json`，阅读版：`evaluation/world_benchmark_160_v1.md`。40一致、80矛盾（时间/人物知识/空间/人物关系/物理规则/世界规则/因果/身份各10）、20相关但证据不足、20完全无关，共192事实、32混合输入、116场景family、44原文来源组。同源变体和开发语义重叠已标记；这不是160个独立设定场景，也没有独立专家标注复核。

运行前验证唯一性、分布、原文/行号/资料摘要、条件和主体标注、短查询token预算。真实通用录入端到端在临时资料副本、独立空存储和索引上运行，只预览；模型复用，不重试，不修改提示或将测试条目写入正式世界观。

每次结果独立保存在 `reports/world_benchmark_160_v1/<运行ID>/`：`cases.jsonl` 逐例原始输出及来源；`summary.json` 状态、冻结实现/提示/依赖锁摘要、指标/分组/混淆矩阵与速度；`report.md` 阅读版。不会覆盖旧baseline或旧benchmark，运行中逐例落盘，只有 `status=complete` 才是完整且配置校验通过的结果。

分别评整体标签和逐事实正确，遗漏及错误进入分母。实体/时间节点归档不能代替事实反馈；额外事实、新结构遗漏、显式主体字段、上下文关键字保留另计。上下文自动计分不证明时间逻辑正确，额外事实不自动等同幻觉，引用真实也不证明原因成立，仍需人工核对。API有效预览与满足全部预期分别报告，延迟不能只看快速失败请求。本轮CPU LLM速度未测。


结构修复 v2 的字段约定、部分失败确认保存与计分版本说明见 [STRUCTURE_FIX_V2.md](STRUCTURE_FIX_V2.md)。

LLM性能指标表、TTFT和显存口径见 [MODEL_METRICS.md](MODEL_METRICS.md)。
