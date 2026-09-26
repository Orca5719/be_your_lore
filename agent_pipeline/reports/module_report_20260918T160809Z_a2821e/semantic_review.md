# Step 48 实际观察：完整报告与过程事件筛选

日期：2026-09-19。离线整理六组保留的真实结果，snapshot_stable=true。这不是正式金标准准确率评测。

## 问题与修改

旧过滤提示允许普通屏幕变化及文件切换独立进入判断；旧报告只是逐条发现列表。filtering-v6增加process_detail，保留重要机制、结果及信息线索，普通过程不独立核对；引用闭包仍保留必要上下文，没有按“屏幕”等词硬删除。

report-v2保留JSON，在前部report内给出总评、矛盾、不确定、吻合、作者确认问题、排除细节及执行问题；compact不再重复平铺findings，完整文件仍保留全部上游原始记录。所有确认建议saved=false，不保存canon。

单靠judge-v5提示仍出现擅自认定右臂的理由，保留中间失败report_mooncity_feedback_fix_20260918.json。judge-v6增加可选uncertainty_code，正常不确定的主要理由由程序按代码生成；原model_reason及raw仍保存。缺失代码兼容回退insufficient；非法代码拒绝并有限重试。旧judge文件整理时不追改历史理由。

## 最终真实CUDA五阶段结果

- report_mooncity_feedback_final_20260919.json：status=ok；液化、进入主机、震中图像三项独立核对，均不确定；屏幕变化及文件闪过两项仅作必要上下文。E1模型未给代码，按兼容规则回退insufficient。完整报告呈现三个问题及两个上下文细节。
- report_right_feedback_final_20260919.json：status=ok；右胸亚巴顿仍为明确矛盾，引用左右心脏及天使宿主原设定；捂胸动作仅作上下文。
- 六组离线整理包含上述两组及四组历史结果；历史上游失败仍partial，不改状态掩盖失败。

## 验证与限制

241项测试通过（2.121秒）；只读审查22项judge及31项report相关测试通过，无重要结构问题。冻结129文件校验不变，archive_ok=true。

安全措辞不证明语义理解或判断准确率提高；原模型理由仍可能胡乱推断，代码选择也来自模型。明确结论的自由文本理由及模型自报适用性仍需后续评测。没有正式Accuracy/Precision/Recall/F1或性能加速结论，没有改旧benchmark、canon或索引。
