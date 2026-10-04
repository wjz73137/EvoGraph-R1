# 四张图片的 API 视觉超图验证

按本地论文第 3.3 节实现：图片生成场景描述、主要物体和可见关系；每条场景/
视觉关系超边均连接 `image::<image_id>` 锚点，再与保留的文本超图共用 GME 空间。
论文没有提供这里使用的视觉原始提示词，因此这是定制 API 实现，不是严格原版复现。

模型仍来自项目 `.env` 的 `GRAPH_LLM_MODEL`，没有切换模型或修改密钥配置。
本轮实际调用 `qwen3.7-flash-2026-07-15` 四次，一图一次，并发 1，最多四次、
每次最多 2048 输出 token，无网络自动重试。请求只有原图字节和固定视觉提示，
不发送文件名、文章标题、Wikipedia 段落或 QA。累计 15360 token；验证和重跑
使用保存的响应，不重复付费请求。输出日志不含密钥，也不重复存图像 Base64。

## 文件与运行

代码：`evograph_mm/kb/visual_scene.py` 定义提示、严格校验和显式审核转换；
`scripts/run_evqa_api_visual_graph.py` 负责抽取、锚点建图和 CPU 索引；
`scripts/audit_evqa_api_visual_graph.py` 检查完整性并记录实际局限；
`scripts/serve_evqa_api_graph.py` 支持本轮独立图和只读场景查询。

数据根目录：
`/home/data/dataset/wjz/EvoGraph-R1/expr_mm/evqa_api_visual_graph_v1/`。
保留清理版父图和更早的原始 API 图，不覆盖它们。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/data/env/wjz/evograph-r1/bin/python \
/home/wjz/projects/EvoGraph-R1/scripts/run_evqa_api_visual_graph.py --extract
```

`--extract` 是付费阶段；四个完成记录已存在时重放缓存，不发新请求。
无效响应停止，不能静默修补。已有 requested/失败网络请求不会自动再发。
审核通过后使用 `--build`，只计算本地 GME 嵌入，不加载本地事实抽取模型：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/data/env/wjz/evograph-r1/bin/python \
/home/wjz/projects/EvoGraph-R1/scripts/run_evqa_api_visual_graph.py --build
```

新增 27 个实体向量、19 个超边向量，复用父图 142 个向量，图片/文档索引
保持原样。全程四线程 CPU，未使用 GPU 或开始训练。已有图不会被覆盖。

## 质量边界与审核记录

四份原始 API JSON 都未直接满足严格 schema：坐标格式不稳定，一份使用了
不同的场景描述字段名且缺少来源标记。原响应保存在 `vision_api_calls.json`，
没有把这些问题包装成“结构化输出全部成功”。

`vision_source_review.json` 保存逐张原图检查后的显式转换，绑定响应 SHA256：

- 排除一个混合单位且无法确定含义的物体框及一条依赖关系。
- 三张图的同尺度 0..1000 框经检查后显式归一化。
- 海面图的两个混合单位框被去掉，水面/天空仍保留整图锚点，不伪造位置框。
- 场景字段名映射和缺少的 image-only 标记通过请求来源审核单独说明，
  不声称原模型输出了缺失字段。
- 删除季节/年龄、未确认树种及不确定小字等推断；保留原始生成内容供回溯。

审核者是代理，不是人工或独立检测器；近似框不等于经检测器验证的区域。
模型置信分数不等于事实正确率。`quality_audit.json` 的原文审核隔离数量与
`report.json` 的低置信度过滤数量是两类统计，不能混为一谈。

最终 118 个实体 = 父图 91 个 + 23 个视觉物体 + 4 个图片锚点。
70 条超边 = 父图 51 条 + 4 条场景 + 11 条视觉关系 + 4 条数据集关联。
关联超边明确标注“图片与文章关联，不是视觉身份识别”。
尤其 Pamban Bridge 的源图只有水面、船只和岸边，看不到桥，不能因为文章
标题有桥就生成视觉桥实体。

同源文章、标签严格归一化同名、GME 相似度门限三者同时满足才合并对象。
本轮没有符合条件的候选（0 个合并），因此仍未完成论文更广泛的实体消歧。
不把字符串近似或通用物体类别等同于具体建筑的身份。

4 条场景自身文本索引检查和 1 条自由文本烟测已通过；这验证连通性和索引，
不是 VQA 准确率。未做全量人工审核、问答评测或策略训练。

## 只读服务

确认本人本任务旧服务正常退出后，在 localhost:8003 服务新图：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/data/env/wjz/evograph-r1/bin/python \
/home/wjz/projects/EvoGraph-R1/scripts/serve_evqa_api_graph.py --port 8003 \
  --working-dir /home/data/dataset/wjz/EvoGraph-R1/expr_mm/evqa_api_visual_graph_v1/E-VQA
```

`POST /search` 检索实体和视觉/文本超边；
`GET /visual-scenes/<image_id>` 返回对应 API 场景、物体、视觉关系和来源限制。
不存在的图像返回 404，在线重载/改图请求返回 405。
回退时使用保留的清理版父图重新启动服务即可，不需要恢复数据或环境。
