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

## 性能优先求解

`test_tsp.py --mode quality` 为每道题保存多种求解器的高质量回路，在边的并集上求解整数规划，再用区域模型选择联合重算的位置和候选新边。区域外已有路径被压缩为必经边，局部求解加入子环约束，最后按原始坐标验证完整回路并保留较短解。大于等于 10,000 城市时，先并行优化空间分区并合并局部回路，再运行批量 2-opt、固定端点的重叠窗口优化和从独立起点运行的 LKH。

性能优先模式需要 SciPy 1.9 或更新版本。课程容器的 Python 3.8 使用独立依赖目录安装 SciPy 1.10.1 和 NumPy 1.24.4，PyTorch 仍来自容器预装环境。GA-EAX 可使用原版可执行程序；以下构建命令另外启用 Large 版本的随机种子和随搜索写出当前路线的功能。Concorde 可选，按上游条款从官网下载 Linux 可执行文件。 [Concorde 下载](https://math.uwaterloo.ca/tsp/concorde/downloads/downloads.htm)

```bash
cd /nfs1/code/project2-quality
export TMPDIR=/nfs1/.tmp
export PIP_CACHE_DIR=/nfs1/.cache/pip
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" /nfs1/conda/envs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python=/opt/conda/bin/python
"$python" -m pip install --index-url https://pypi.org/simple \
  --target /nfs1/conda/envs/project2-milp scipy==1.10.1 numpy==1.24.4
export PYTHONPATH=/nfs1/conda/envs/project2-milp
"$python" build_quality_eax.py /nfs1/code/vendor/GA-EAX/GA_EAX_1.0/Large \
  /nfs1/code/vendor/GA-EAX/eax-quality

data=/nfs1/datasets/neural-tsp/quality-v1
weights=/nfs1/checkpoints/neural-tsp/quality-v1
common=(--lkh /nfs1/code/vendor/LKH3/LKH --scratch "$TMPDIR" \
  --eax /nfs1/code/vendor/GA-EAX/eax-quality \
  --neurolkh /nfs1/code/vendor/NeuroLKH \
  --static /nfs1/checkpoints/neural-tsp/pilot-v2/static.pt)
"$python" -m neural_tsp.train_regions prepare "${common[@]}" \
  --training --seed 630000 --output "$data/reproduce-train"
"$python" -m neural_tsp.train_regions prepare "${common[@]}" \
  --training --seed 631000 --output "$data/reproduce-validation"
"$python" -m neural_tsp.train_regions fit --training "$data/reproduce-train" \
  --validation "$data/reproduce-validation" --output "$weights/reproduce-first.pt"
"$python" -m neural_tsp.train_regions prepare "${common[@]}" --training \
  --seed 632000 --mine "$weights/reproduce-first.pt" --output "$data/reproduce-mined"
"$python" -m neural_tsp.train_regions fit \
  --training "$data/reproduce-train" "$data/reproduce-mined" \
  --validation "$data/reproduce-validation" --start "$weights/reproduce-first.pt" \
  --output "$weights/reproduce-region.pt"
```

`prepare` 使用合成实例，按实例种子保存标签。训练和验证使用不同种子。训练可加入强求解器的其他路线、留掉最强父路线后的状态及少量接近收敛的扰动状态；留出集生成时省略 `--training`，保留标准的父路线重组状态。`benchmark` 固定各方法可选区域和区域求解结果，并按实例规模与分布汇报相对父路线的长度变化。

```bash
"$python" -m neural_tsp.train_regions prepare "${common[@]}" \
  --seed 633000 --output "$data/reproduce-holdout"
"$python" -m neural_tsp.train_regions benchmark --data "$data/reproduce-holdout" \
  --model "$weights/reproduce-region.pt" \
  --output /nfs1/outputs/neural-tsp/quality-v1/reproduce-ablation.json
"$python" test_tsp.py --mode quality --dataset "$data/course/heldout1000.txt" \
  --lkh /nfs1/code/vendor/LKH3/LKH --scratch "$TMPDIR" \
  --eax /nfs1/code/vendor/GA-EAX/eax-quality \
  --neurolkh /nfs1/code/vendor/NeuroLKH \
  --concorde /nfs1/code/vendor/Concorde/concorde \
  --checkpoint /nfs1/checkpoints/neural-tsp/pilot-v2/static.pt \
  --region-checkpoint "$weights/region.pt" --seconds 30 --quality-seeds 4 \
  --quality-passes 0 --quality-regions 48 --quality-attempts 12 \
  --exact-seconds 180 --output /nfs1/outputs/neural-tsp/quality-v1/reproduce-course
```

质量模式的 `--seconds` 是每个父路线求解调用的时长设置，合并、局部及全图精确求解分别有时长参数。默认 `hybrid` 策略同时使用模型和父路线分歧选择区域，停滞时再尝试剩余区域。`--quality-passes 0` 在连续两轮未缩短路线时退出。`incumbents/` 保存运行中的当前最好路线，使用相同配置加 `--resume` 能从中断的实例继续。评测报告课程 `avg_aug_gap`、配对基线胜率、路线长上下界及证明状态。并集和区域的最优状态对应其整数化子问题。全局距离界使用向下修正后的全图下界与向上修正后的回路上界。`exact_optimal` 只在上下界闭合或精确整数动态规划排除所有更短的替代路线时成立。
