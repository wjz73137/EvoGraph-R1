# 本地 E-VQA GLDv2 最小链路

所有模型/数据/输出位于数据盘。原始 QA 和已下载图片只读。
不使用 API Key，不安装依赖，不调用 Ray，不启动训练，不生成 mock 索引。

## 实验范围

- `run_evqa_retrieval_smoke.py`：固定 `paper_64_16_seed0`，从校验过的官方
  KB 抽取 82 篇对应文章的全部非空段落，形成 1294 个重叠文本块。
  真实 GME 图文查询 → FAISS → 本地 Qwen2.5-VL-3B 生成，测试固定 16 条。
- 文档仅使用百科正文；查询仅使用问题及图片。标准答案和目标百科 URL
  只在查询/生成完成后用于诊断评估。未修改原始 CSV 的 evidence 字段。
- 这是 **82 篇文章的受限检索库**，不是论文完整 benchmark。
  简单别名/多答案词组匹配不是官方 BEM 分数。
- `run_evqa_native_graph_smoke.py`：只使用现有训练集的前四篇不同文章，
  每篇取首个非空段落的首个 1200 字符文本块，建立原生 GraphR1 超图。
  这是四段落结构验证，不是完整 64+16 知识图，也不证明事实提取准确。

## 环境兼容

`evograph_mm/kb/indexing/gme_compat.py` 使用已安装 Transformers 5 原生
Qwen2-VL 模块加载官方 GME 权重，检查缺失/额外/不匹配权重，保留官方提示、
顺序位置、末尾 token 池化及归一化。没有编辑模型快照或第三方库。
尚未与旧版本实现做数值逐项等价评估。

原生超图保留全部本地 LLM 原始输出。若记录末尾的数字分数前出现 `|>`，
仅补回缺少的 `<`，写入单独解析输出及修复计数；不新增或改写事实记录。

## 运行

重跑前必须检查目标 GPU；脚本发现该卡已有计算进程或空闲显存不足就退出，
不会停止其他进程。使用绝对路径解释器，单卡，CPU 线程最多 4。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/data/env/wjz/evograph-r1/bin/python \
/home/wjz/projects/EvoGraph-R1/scripts/run_evqa_retrieval_smoke.py --gpu 0
```

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/data/env/wjz/evograph-r1/bin/python \
/home/wjz/projects/EvoGraph-R1/scripts/run_evqa_native_graph_smoke.py --gpu 0
```

脚本有独占锁、输出归属/输入哈希检查及进度记录。只接受同配置的自有输出，
不清理已有目录。检索脚本恢复已提交的 embedding 和已生成的完整测试记录；
原生建图脚本复用原始 LLM 输出和已持久化的源文档。

## CPU 检索服务

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/data/env/wjz/evograph-r1/bin/python \
/home/wjz/projects/EvoGraph-R1/scripts/run_evqa_native_graph_smoke.py \
--serve --port 8003
```

仅监听 `127.0.0.1:8003`，单进程、CPU 模型；仅允许读取和 POST `/search`。
模型约占 9 GB 主存，不占 GPU。建图完成报告必须通过才能启动。
`/search` 保留原生协议，返回 JSON 字符串列表。四张卡上的其他任务不受影响。

```bash
curl --noproxy '*' -s http://127.0.0.1:8003/status
```

日志与 PID：`/home/data/dataset/wjz/EvoGraph-R1/logs/evqa_*`。
输出：`/home/data/dataset/wjz/EvoGraph-R1/expr_mm/evqa_gme_rag_64_16_seed0/`
及 `expr_mm/evqa_native_graph_smoke/E-VQA/`。

正式 GLDv2 派生集仍有 8 张图片不可用：1951 条可用 QA（1891 train、60 test），
不是完整的 1898+61。此最小链路没有重新采样或改变正式子集。
