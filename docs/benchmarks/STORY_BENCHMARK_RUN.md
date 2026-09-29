# Story-Level Benchmark：检查结果与运行

当前可运行的是4段pilot/标注草案，不是最终24段。完整目标仍缺20段故事及金标；现有金标尚未声称用户已审核。
已补运行器story_benchmark.py、计分模块story_benchmark_metrics.py及11项回归测试。最终112项测试通过，数据/来源/索引快照校验通过。
本轮没有执行真实GPU模型评测；运行与计分命令链路用模拟模型验证，不能将测试成绩当模型准确率。

## 直接试跑

```powershell
Set-Location "C:\Users\Xhang\Desktop\project\worldcheck"
& ".\.venv\Scripts\python.exe" -X utf8 ".\story_benchmark.py" run --device cuda
```

默认1次完整流程预热，之后跑4段故事，Qwen/BGE/索引各加载一次并复用。--device cpu或auto也可用。
不需重建索引，不写canon，不更改冻结Fact-Level。新结果保存到自动生成的独立reports/story_benchmark_时间_编号目录，终端会显示完整路径。不会覆盖旧报告。
--output可指定尚不存在的新目录；已有目录会拒绝覆盖。--warmup 0可关闭预热，但不会将测量标成预热后时延。

只检查资料、不加载模型：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 ".\story_benchmark.py" validate
```

## 这轮记录的指标

检测指标已经从单个总Accuracy扩展为以下独立项，不能用其中一项替代其他项：

- 事实提取Precision、Recall、F1，反映误提和漏提。
- 事实级冲突检测Precision、Recall、F1，反映冲突定位；漏提/判断失败造成的未报冲突进入FN。
- 故事级冲突检测Precision、Recall、F1，只表示发现故事有冲突，不代表定位正确。
- 三分类事实正确率及混淆计数，一致/矛盾/不确定分别保留；遗漏/失败算错。
- 核心证据Recall@K（默认5），分母包含漏提/失败，多证据组合须全部命中。
- 日常段落合法空提取率、误提数、阶段失败、待复核、重试及格式规范化次数。

分母0使用N/A，不当100%。检测指标依赖人工一对一语义配对，不能按完全相同的文字或Qwen自己的标签自动评分。

性能自动记录Model/Parameters/Precision、模型显存/张量占用、请求峰值allocated/reserved显存、生成调用TTFT中位数、加权decode tok/s、完整故事时延中位数/P95、顺序吞吐及平均输入tokens。
模型/BGE加载单列；显式预热不计检测和测量行。所有测量请求保留，包括失败和重试。TTFT是单次LLM生成计时，不是用户界面首次反馈；平均输入tokens也按LLM调用，不是按故事。
GPU精度仍为Qwen NF4权重/BF16计算、BGE FP32，不冒充FP16。显存是PyTorch进程计数，峰值包含Qwen+BGE，reserved可能保留预热缓存；CPU显存显示Not recorded。首请求含加载的完整用户时延未测，不从其他数值推算。
四段pilot的P95样本很少，性能值只用于开发检查。若预热失败会单独记录，普通测量时延仍保留，不虚称成功预热。

## 输出文件

- dataset.json：本次实际使用的测试集快照。
- raw.jsonl：每段故事结束即保存，包含提取、候选筛选、检索、判断、原始回复及调用计时；失败也保存，其他段继续。
- report.json：模型/依赖/设备配置、模型及提示版本摘要、加载/显存、阶段状态及重试/规范化统计。
- warmup.json：预热结果，与测量分开。
- performance.json、benchmark.md：性能统计及可读表。Accuracy和检测P/R/F1在语义配对前待计分。
- review.md：故事、金标、预测事实及必要上下文，供审阅。
- review.json：一对一语义配对模板，默认reviewed=false。
- metrics.json：完成配对后执行score才生成的检测指标。

## 运行之后

将结果目录告诉我，我们按原文和设定审阅配对及错误；不必自己猜怎样配对。
每案review.json的matches填写形如{"prediction_id":"F2","gold_id":"G1"}的条目；只配对语义等价事实，忽略Qwen判断是否正确。确认一案的所有预测/gold已审阅才设reviewed=true。
未配对预测算FP，未配对gold算FN，所以不能把空模板直接全部设true当作已审核。纯日常或彻底失败的案例也需审阅，但可合法保留空matches。
审核完成后计分，不再调用模型：

```powershell
& ".\.venv\Scripts\python.exe" -X utf8 ".\story_benchmark.py" score "结果目录的完整路径"
```

审核与dataset/raw通过run_id及SHA256绑定，不能混用另一轮结果；资料/代码/提示运行时变化会拒绝计分。所有原始输出保持不变。
生成的分数仍标为pilot/draft；不能宣称完整24段正式benchmark或独立人工审核的最终金标成绩。
