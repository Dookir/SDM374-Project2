# Project 2 · 稀疏神经候选边与 LKH

公开实验见 [RESULTS.md](RESULTS.md)。评测入口可用 `--resume` 延续同配置、同顺序的中断任务。

输入是 `[N, 2]` 城市坐标，输出是从 0 编号的闭环城市排列。稀疏图包含双向合并的 32 近邻边、Delaunay 边以及保底回路。六层、宽度 128 的图网络先预测静态边分数。初始路线可用神经排序构造、LKH 的 Sierpinski 构造，或者两者择优。搜索阶段保留当前路线边及几何近邻，用固定的三种候选排序产生边组，由修复头预测每组给 LKH 带来的缩短量。每轮以原始浮点坐标验收，保留最短合法回路。

## 服务器目录与依赖

在 `/nfs1/code/project2-neural-tsp` 运行。使用课程容器预装的 `/opt/conda/bin/python`，实测 PyTorch 1.13 和 SciPy 1.6。LKH 使用 c4v4/LKH3 的 `5bd63680970445d4a595ca2c966ac593b0b32ad1`，服务器编译位置为 `/nfs1/code/vendor/LKH3/LKH`。EAX 和 NeuroLKH 为可选对照。可传入已安装的 Concorde 生成额外教师标签。各外部程序遵循原项目的使用条款。

```bash
export CONDA_PKGS_DIRS=/nfs1/conda/pkgs
export PIP_CACHE_DIR=/nfs1/.cache/pip
export TMPDIR=/nfs1/.tmp
export MPLCONFIGDIR=/nfs1/.cache/matplotlib
export PYTHONDONTWRITEBYTECODE=1
cd /nfs1/code/project2-neural-tsp
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
lkh=/nfs1/code/vendor/LKH3/LKH
dataset=/nfs1/datasets/neural-tsp/mixed
checkpoint=/nfs1/checkpoints/neural-tsp/mixed
public=/nfs1/datasets/ttpl/data/TSP/test/MCTS_tsp1000_test_concorde.txt
dev=/nfs1/datasets/ttpl/data/TSP/val/validation_TSP1000_n128.txt
```

## 训练与评估

```bash
/opt/conda/bin/python -m neural_tsp.train prepare \
  --lkh "$lkh" --scratch "$TMPDIR" --data "$dataset" \
  --examples 240 --validation 20 --teacher-runs 2 --label-seconds .6 \
  --eax /nfs1/code/vendor/GA-EAX/GA_EAX_1.0/Normal/jikken --extra-teacher-seconds 2 \
  --exclude /nfs1/datasets/lehd/test_TSP1000_n128.txt "$public" "$dev"
/opt/conda/bin/python -m neural_tsp.train fit \
  --lkh "$lkh" --scratch "$TMPDIR" --data "$dataset" --output "$checkpoint" \
  --static-epochs 4 --repair-epochs 3

/opt/conda/bin/python test_tsp.py --dataset "$public" --lkh "$lkh" --scratch "$TMPDIR" \
  --checkpoint /nfs1/checkpoints/neural-tsp/pilot-v2/static.pt --mode static --seconds 2.16 \
  --baseline-jsonl /nfs1/outputs/neural-tsp/lehd-ttpl/routes.jsonl \
  --output /nfs1/outputs/neural-tsp/ttpl-static-repeat
```

生成器将同一实例的变体和搜索状态保存在同一文件。可通过 `--labels` 加入带参考路线的训练文件。生成前用 `--exclude` 指定评测文件以检查坐标重复。短时间搜索可能沿用已取得的合法回路，评估报告同时记录预算和实测耗时。EAX 和 NeuroLKH 超时且没有产出路线的实例单独计数，用 `complete` 标记整组是否完整。

`test_tsp.py` 支持 TTPL 的数据文件与 `--problem_size`、`--test_in_tsplib True` 用法。合成集 Gap 使用两组平均长度的比值，TSPLib 使用逐题 Gap 的平均值，输出都用百分数。可选 `--augmentations 8` 会在原始坐标上择优，并在八个视图间分配总搜索预算。TTPL 自带的投影和八视图打分合并由原始基线程序执行。

`python -m neural_tsp.lehd_baseline` 支持从原版 LEHD 或课件 TTPL 保存逐题路线。将其 `routes.jsonl` 传给 `test_tsp.py --baseline-jsonl` 即可计算严格胜率。使用 `run_experiments.sh DATASET CHECKPOINT OUTPUT_DIR BASELINE_SECONDS_PER_INSTANCE` 运行 1×、4×、16× 时间曲线和初始化、原生候选、随机选择消融。将 `PROJECT2_BASELINE` 指向同一数据集的基线路线即可做逐题配对。

## 来源

- [课程 TTPL](https://github.com/CIAM-Group/TTPL/tree/main/TTPL/lehd/TSP)
- [LEHD](https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD)
- [NeuroLKH](https://github.com/liangxinedu/NeuroLKH)
- [LKH3](https://github.com/c4v4/LKH3)
- [GA-EAX](https://github.com/nagata-yuichi/GA-EAX)
