# E-VQA 增补抽取方案

这是文本侧建图前的抽取诊断，不是完整多模态建图、检索或训练复现。
输入与上一轮固定四篇 Wikipedia 段落完全相同，不读取 QA 问题和答案。
使用已下载的 Qwen2.5-VL-7B，不下载模型、不修改依赖。

| `--arms` | 方案 | 推理步骤 |
| --- | --- | --- |
| `inventory_7b` | 事实清单引导 | 先列带引用的覆盖清单，再抽取候选事实 |
| `entity_7b` | 实体优先 | 先列实体及别名，再抽取带明确主体的关系陈述 |
| `review_7b` | 二次复核 | 读取上一轮逐句候选，对照原文纠错、补漏 |
| `window_7b` | 相邻句窗口 | 两句一窗、步长一句；全文仅作指代上下文 |

不同方案输出在数据盘 `expr_mm/evqa_extraction_comparison_advanced_v1/`，
与之前的图和实验目录隔离。`calls.json` 保存每次提示词、原始响应、
token 数和时间；`report.json` 保存候选、引用字符位置和拒绝原因。
`owner.json` 固定输入、代码和复核草稿哈希，不覆盖配置不同的产物。

启动前检查 `nvidia-smi`，用户确认 GPU 后再运行。下面两组是独立的单卡
进程，不是多卡训练；仅在 GPU 2、3 空闲时适用。程序还会检查显卡当前
没有计算进程且至少有 20000 MiB 空闲显存，每个进程限制为四个 CPU 线程。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/data/env/wjz/evograph-r1/bin/python \
/home/wjz/projects/EvoGraph-R1/scripts/compare_evqa_extraction_advanced.py \
--arms inventory_7b entity_7b --gpu 2
```

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/data/env/wjz/evograph-r1/bin/python \
/home/wjz/projects/EvoGraph-R1/scripts/compare_evqa_extraction_advanced.py \
--arms review_7b window_7b --gpu 3
```

复核组需要上一轮 `evqa_extraction_comparison_v2/sentence_7b/report.json`。
复核和清单标签也是模型输出，不是人工证明，不能作为真值。
程序只校验引用、数字、目标范围和残句；语义验证仍标记为 false。
仅做完全相同 statement/evidence 的去重，不保证语义去重。
实体清单是抽取提示信息，不会创建生产图节点或超边。

所有组结束后，通过 CPU 汇总八组并校验其原文输入一致性：

复核组初次有三篇输出为 JSON 数组而非对象。先用 CPU 将数组包装为
`facts`，不改动任何候选陈述或引用。严格协议原始结果保留在 `review_7b`，
格式统一结果另存 `review_7b_array_normalized`；汇总保留这三次协议偏差。

```bash
/home/data/env/wjz/evograph-r1/bin/python \
/home/wjz/projects/EvoGraph-R1/scripts/revalidate_evqa_review.py
```

```bash
/home/data/env/wjz/evograph-r1/bin/python \
/home/wjz/projects/EvoGraph-R1/scripts/summarize_evqa_extraction_advanced.py
```

引用覆盖率不等于事实召回率；候选条数不等于正确事实数。
复核组耗时不包含上一轮生成草稿的耗时，不宜直接比较端到端效率。
本实验不更换已有图、不改变 8003 服务，也不启动训练。
