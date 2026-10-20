# 5B 实现验证（2026-10-20）

- 74项相关测试通过：5A、5B和4.2历史协议/推测解码回归。测试以fake模型替代实际GPU生成，走原窗口、修复、队列、字段校验和审核路径。
- validate-b本机通过：固定Qwen本地缓存、Transformers4.57.6、冻结数据/代码和索引；检查时C盘及TEMP约30GiB空闲。它不确认真实生成速度或准确率。
- TDD发现并修复：队列清空last_batch_generation时遮蔽底层遥测、完成组缺文件被跳过、下游清单未绑定、失败组日志检查遮蔽失败状态。
- 独立只读审查的两项P2已修复并复核：已有成功故事的原始wire/推理日志不能丢失后续跑；组内manifest、父实验摘要、故事、row.system、review.arm和下游arm绑定，防止复制目录造成串组。
- 初提、条件、否定、跨句actor、非空心理字段保留；短协议不猜modality。生成tokens来自真实回复，程序补齐的JSON不计生成tokens。
- 本轮不安装依赖、不修改Windows/WSL配置、不复制/导出模型、未运行GPU smoke或正式benchmark。
- 原生产pipeline文件未改；5B仍是待用户运行的实验候选。长实验命令见README。
