# E-VQA 四方案事实抽取对照

输入固定为原有超图冒烟实验的四篇 Wikipedia 段落，读取其 `owner.json`；
不读取问题、答案或 QA evidence 标签。使用已下载的 Qwen2.5-VL-3B/7B。

| 参数 `--arm` | 模型 | 处理方式 |
| --- | --- | --- |
| `whole_3b` | 3B | 整段抽取 JSON 事实及引用 |
| `whole_7b` | 7B | 相同提示词，整段抽取 |
| `sentence_7b` | 7B | 逐句抽取，全文仅用于指代消解 |
| `extractive_7b` | 7B | 直接选取原文片段，不采纳模型改写 |

运行前检查 `nvidia-smi`，并由用户确认 GPU。不同方案是独立的单卡进程；
程序通过 `CUDA_VISIBLE_DEVICES` 隔离，只接受当前无计算进程且至少有
20000 MiB 空闲显存的卡，每进程 CPU 线程数为 4。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/data/env/wjz/evograph-r1/bin/python \
/home/wjz/projects/EvoGraph-R1/scripts/compare_evqa_extraction.py \
--arm sentence_7b --gpu 2
```

产物在数据盘 `expr_mm/evqa_extraction_comparison_v1` 与 `v2`。
v1 保留首次四卡并行的全部原始结果。v2 修复括号内缩写分句、过滤末尾残句，
并将仅空白不同、且唯一匹配的引用映射回原文精确字符区间。
整段三组通过 CPU 重验 v1 原始模型输出；分句组重新推理。
所有原始提示词与模型输出保留在对应推理目录的 `calls.json`。

出处校验包括引用原文位置、额外数字、末尾残句、目标句范围和完全重复项。
这些检查不能判定语义蕴含。`semantic_entailment_verified` 保持 false。
原文引用方案不采纳改写后的 statement，仅保存原始片段与文章上下文。
当前结果未写入已有生产图，也未更换 8003 服务。

```bash
/home/data/env/wjz/evograph-r1/bin/python \
/home/wjz/projects/EvoGraph-R1/scripts/summarize_evqa_extraction.py
```

统计中的“证据字符覆盖率”是完整源句非空白字符被引用的比例，
不是事实召回率、回答准确率或论文分数。一次引用整段可以有很高的
证据覆盖率，但对应的改写 statement 仍可能遗漏事实或产生错误。
