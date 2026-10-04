# API 抽取小规模图基线

当前知识抽取调用项目 `.env` 的 `GRAPH_LLM_MODEL`，地址与密钥由
`OPENAI_BASE_URL` 和 `OPENAI_API_KEY` 提供，不加载本地抽取模型。
不修改 `.env`，不发送 QA 答案；只有原文和原始抽取示例传给配置的 API。
使用项目原始 `graphr1/prompt.py`，以及原始默认的两轮补漏。
当前配置是千问 API，不是论文的 GPT-4o-mini，属于 API 替代基线。

五个旧本地模型图已从工作目录移入任务数据盘 `.trash/local-extraction-*`。
回收区的 `manifest.json` 记录原路径；这是可恢复移除，不是永久擦除。
删除原始数据、模型或历史非图诊断不在本次范围内。
原始四篇文档和出处保存在新目录的 `source_snapshot.json`，不复用旧图事实。
`scripts/retire_local_extraction_graphs.py` 只允许五个确认过的本地抽取目录，
先验证旧服务进程属于当前用户且命令匹配，再发送 SIGTERM。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/data/env/wjz/evograph-r1/bin/python \
/home/wjz/projects/EvoGraph-R1/scripts/run_evqa_api_graph.py
```

这一步执行真实付费 API 调用，最多 20 次不同请求，每次最多 4096 输出 token，
并发为 1，不自动重试网络失败。响应、token 用量和结束原因记录在
`expr_mm/evqa_api_graph_baseline/E-VQA/api_calls.json`；不记录密钥。
空输出、截断和配置变更会停止；相同成功请求可重放缓存，避免重复付费。

提取完成后构建 GME 检索索引。GME 是嵌入模型，不做实体事实抽取；
本轮使用 CPU、四线程，不占 GPU：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/data/env/wjz/evograph-r1/bin/python \
/home/wjz/projects/EvoGraph-R1/scripts/run_evqa_api_graph.py --index
```

图、索引及原文存放在数据盘
`/home/data/dataset/wjz/EvoGraph-R1/expr_mm/evqa_api_graph_baseline/`。
汇总 `report.json` 区分抽取完成、索引完成和检索测试。
所有图元素映射到真实原文段落，不意味着每个模型生成的事实都正确。
`factual_consistency_verified` 保持 false，不以条数当作准确率。
当前范围只有四段文本，图片只做关联与嵌入，未完成视觉场景事实抽取。

索引确认就绪、8003 无其他监听程序后，可启动 API 图的只读检索服务：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/data/env/wjz/evograph-r1/bin/python \
/home/wjz/projects/EvoGraph-R1/scripts/serve_evqa_api_graph.py --port 8003
```

## 原文核对后的独立清理版

`scripts/clean_evqa_api_graph.py` 的默认运行只做预演；`--apply` 才创建
`expr_mm/evqa_api_graph_cleaned_v1/E-VQA/`，不会覆盖原始 API 基线。
计划限定到已固定哈希的四篇文档和具体记录，不适合直接应用到新数据集。

本轮隔离 6 条超边和 3 个实体，包括未完成句子、原文未出现的具体地名、
拆散的“可能 A 或 B”判断，以及待复核的艺术中心组成关系。
隔离不表示它们必然为假，只表示当前原文不足以支持作为独立事实返回。
合并 9 条逐项核对过的重复表述，保留复合事实与单独事实的区别，
合并时保留邻接关系和原文出处，不因重复而提高置信度。
命名建筑、桥梁及其别名统一为 LOCATION，是显式 schema 约定，不是准确率结论。
日期 1571 标明是由 1572 减一年推导；建造者描述保留完整可能性和析取范围。

原始图、原始 API 输出保持不变；`cleanup_audit.json` 保存隔离和合并前的记录、
图节点、关联边与出处。清理图含 91 个实体、51 条超边，所有问题记录也从活动
实体/超边索引中排除，避免只软删除超边后仍能检索到关联实体。
复用 139 条原向量，只用四线程 CPU 重算 3 条修改过的实体描述；
图片和文档索引不变，无新的 API 调用，也不加载本地事实抽取模型。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/data/env/wjz/evograph-r1/bin/python \
/home/wjz/projects/EvoGraph-R1/scripts/clean_evqa_api_graph.py --apply
```

已有完成目录会被拒绝覆盖。只有在未完成目录的活动记录、图、出处和基线哈希
均与当前计划一致时，才允许显式 `--apply --resume` 续跑索引/验证。
汇总在清理版根目录 `report.json`；本轮做了 4 组真实 CPU 检索测试。
这仍是抽取质量清理和检索连通性验证，没有全量人工事实审核或问答准确率评测；
也没有视觉场景事实抽取、策略训练或训练就绪结论。

确认旧服务是本人本任务进程且已正常退出后，可在 8003 启动清理版只读服务：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/data/env/wjz/evograph-r1/bin/python \
/home/wjz/projects/EvoGraph-R1/scripts/serve_evqa_api_graph.py --port 8003 \
  --working-dir /home/data/dataset/wjz/EvoGraph-R1/expr_mm/evqa_api_graph_cleaned_v1/E-VQA
```

服务只允许基线或清理版两个已知目录，仍只监听 localhost，禁止在线改图和重载。
回退时停掉确认归属的本任务服务，再用默认命令启动保留的原始 API 基线即可。
