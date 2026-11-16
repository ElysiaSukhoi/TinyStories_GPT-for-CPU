# Tiny GPT CPU

具身智能基础作业三项目。目标是在 Windows CPU 上从随机权重预训练小参数 GPT，再进行英文故事续写微调。

## 检查本机环境

在仓库根目录运行以下 PowerShell 命令，直接使用项目 Conda 环境里的 Python。

```powershell
& .\.conda\python.exe scripts/check_environment.py
```

脚本验证 CPU 版 PyTorch 的前向、反向和 AdamW 参数更新，同时训练一个微型 byte-level BPE 分词器，检查编码解码能否还原原文。

## 在 Windows 复现环境

从 Anaconda 的 base 环境在仓库根目录运行。依赖版本记录在 [requirements.txt](requirements.txt)，PyTorch 使用官方 CPU 构建。

```powershell
python -m conda create --prefix .\.conda --override-channels --channel https://conda.anaconda.org/conda-forge python=3.12.16 pip --yes
& .\.conda\python.exe -m pip install --only-binary=:all: -r requirements.txt
& .\.conda\python.exe -m pip check
& .\.conda\python.exe scripts/check_environment.py
```

## 准备 TinyStories 数据

在仓库根目录运行。缺少的 Parquet 分片会从 Hugging Face 下载到本地缓存，程序会核对 SHA-256，再生成 tokenizer 和训练数据。

```powershell
& .\.conda\python.exe scripts/prepare_data.py
```

默认种子为 42，原始分片缓存于 `data/raw`，输出位于 `data/prepared`。复建时可用 `--output-dir data/rebuild` 选择新目录并复用缓存。

使用 TinyStories 固定版本的第一个训练分片及完整官方验证分片。训练分片随机抽取 65,000 个候选行，按 Unicode、空白和大小写归一化后的故事哈希去重。先保留官方验证故事，再筛除命中这些哈希的训练候选。预训练和 SFT 使用互斥的故事，SFT 的三份数据还对完整开头去重。 [TinyStories 数据卡](https://huggingface.co/datasets/roneneldan/TinyStories) · [固定版本的分片与下载入口](https://huggingface.co/datasets/roneneldan/TinyStories/tree/6d02f92396f7e3e60c2251070b6e0b168568074b/data)

| 数据集 | 故事数 | Token 数或样本长度 |
| --- | ---: | ---: |
| pretrain_train | 32,000 | 7,110,586 个 token |
| pretrain_val | 1,500 | 323,458 个 token |
| sft_train | 4,000 | 每条 55–128 个 token |
| sft_val | 400 | 每条 57–113 个 token |
| sft_test | 400 | 每条 61–128 个 token |

- `tokenizer.json` 仅在预训练训练集上训练，词表含 4,096 个 token。`manifest.json` 记录数据版本、分片校验值、种子、去重统计、各集合规模和 BOS、EOS、PAD 的 ID。
- `pretrain_train.bin` 和 `pretrain_val.bin` 是小端 `uint16` 的连续 token 缓存。每篇故事以 BOS 开头、EOS 结尾，对应的 JSONL 保留故事和 ID。
- SFT JSONL 包含 `story_id`、`instruction`、`input` 和 `output`。取前两句为开头，接下来的两句为续写。分句采用保守规则，跳过带双引号对话、常见缩写或其他无法稳定切分的开头。`scripts/story_task.py` 提供共享提示模板及独立的提示和答案编码。长度限制 128 包含 BOS、提示、答案和 EOS。

## 验证 GPT 与小批训练

`model.py` 的默认配置为 4 层、4 个注意力头、192 隐藏维度和 128 token 上下文，共 2,590,848 个参数。模型将词嵌入与学习式位置嵌入相加，依次经过由因果多头注意力、Pre-LayerNorm 和 GELU 前馈层组成的残差块。语言模型输出头与词嵌入共享权重。 [架构与训练参考章节](https://github.com/rasbt/LLMs-from-scratch#table-of-contents)

准备 TinyStories 数据后，从仓库根目录运行以下命令。

```powershell
& .\.conda\python.exe -m scripts.check_model
```

脚本从预训练缓存取两条固定的 64 token 输入，每个输入位置以上游序列中的下一个 token 为标签。它会扰动输入后半段以检查因果掩码，然后用交叉熵和 AdamW 更新 120 次，保存并重新加载 `runs/model-check/overfit.pt`。最后核对保存前后的 logits，并在 CPU 上按贪心策略逐 token 生成续写。

本机以 4 个计算线程运行时，固定小批的交叉熵从 8.31687 降到 0.12867，下一 token 准确率为 96.9%。重新加载后生成的 20 个 token 与该训练片段中的续写一致。这组结果用于检验模型能否拟合固定训练样本。

## CPU 基准与预训练

基准使用真实训练语料，每档先预热 12 步，再计时 50 次 AdamW 更新。每次更新由两个 8 × 128 token 的微批累积而成，共 2,048 个 token。峰值为整个 Python 进程的工作集。验证单独计时，使用固定的 256 个验证窗口，共 32,768 个 token。

```powershell
foreach ($threadCount in @(4, 8, 16)) {
    & .\.conda\python.exe train_pretrain.py --benchmark --threads $threadCount --steps 50
}
```

| PyTorch 线程 | 训练 token/s | 峰值内存 MiB | 单次验证 s |
| ---: | ---: | ---: | ---: |
| 4 | 13,575 | 492.2 | 0.88 |
| 8 | 13,043 | 492.6 | 0.74 |
| 16 | 10,715 | 498.3 | 0.87 |

本机按吞吐选择 4 线程。预训练从随机权重开始，使用 FP32 和 AdamW。前 100 步将学习率线性升到 0.0003，后续保持该值。梯度范数上限为 1.0，训练窗口使用固定种子有放回抽样。

```powershell
& .\.conda\python.exe -u train_pretrain.py --threads 4 --steps 1000 --eval-every 250
& .\.conda\python.exe -u train_pretrain.py --threads 4 --steps 4000 --eval-every 250 --resume
```

`--steps` 是累计的优化器更新次数。用 `--run-dir` 指定新目录可从头另跑一组实验。默认输出位于 `runs/pretrain`。

- `last.pt` 保存当前权重、优化器、批次与 PyTorch 随机状态、历史指标和当时最佳权重。续训会核对配置与训练数据指纹。
- `best.pt` 导出验证交叉熵最低的一步，用于推理和后续 SFT。
- `metrics.json` 记录随机初始化基线与每轮评估的交叉熵、验证困惑度、训练累计时长和进程峰值内存。
- `samples.json` 保存三个验证集故事开头在随机初始化和最佳预训练权重下的续写。两组样本使用相同的提示、种子、温度 0.8、top-k 20 和 48 token 生成上限。

### 本机预训练结果

| 优化器更新 | 累计训练 token | 固定窗口验证交叉熵 | 验证困惑度 |
| ---: | ---: | ---: | ---: |
| 0（随机初始化） | 0 | 8.3650 | 4,294.16 |
| 1,000（试跑结束） | 2,048,000 | 3.6824 | 39.74 |
| 4,000（最佳权重） | 8,192,000 | 2.8928 | 18.04 |

两次命令合计约 13 分钟，其中训练主体 757.1 秒，平均约 10,821 token/s，实际训练的峰值工作集约 523 MiB。最佳权重另在验证缓存的全部完整窗口上复核，计分 323,456 个下一 token，交叉熵为 2.9060，困惑度为 18.28。对应随机基线的困惑度为 4,298.87。

`full_validation.json` 记录完整窗口复核，`loss_curves.png` 是基于固定窗口指标绘制的损失与困惑度曲线。续写样例中仍可见重复的开头句式和人物跳转，后续将用留出的两句续写测试集评估 SFT 的效果。

## 监督微调与同题比较

用 `runs/pretrain/best.pt` 初始化全部参数，使用 4 个 CPU 计算线程、批大小 16 和 3 轮 AdamW 微调。学习率前 25 次更新线性升至 `1e-4`，随后保持该值。批次右侧填充 `PAD`，位移后仅续写及其 `EOS` 参与交叉熵。第一个续写 token 由提示的末位预测。每轮在同一份 400 条 SFT 验证集上按目标 token 加权计算交叉熵，选损失最低的轮次。 [监督微调参考章节](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07)

```powershell
& .\.conda\python.exe -u train_sft.py
& .\.conda\python.exe -u train_sft.py --evaluate
```

训练产物为 `runs/sft/best.pt` 和 `runs/sft/training_metrics.json`，复跑可用 `--run-dir` 指定新目录。测试命令在冻结的 400 条测试样本上比较预训练与 SFT 权重。两个模型使用同样的提示、贪心解码、64 个新增 token 上限、128 token 上下文和 EOS 停止。两句率用与数据构造共享的保守分句规则统计完整生成文本，含双引号对话、未完成的第三句等情况会判为失败。另统计 EOS 停止率与同时满足两项的比例。

`test_metrics.json` 保存汇总指标与测试集、权重、分词器的 SHA-256。`test_outputs.jsonl` 逐题保存参考续写、两组生成文本及判定。

### 本机微调结果

三轮共 750 次更新，耗时约 92 秒。SFT 验证集的答案交叉熵从 3.43655 降至第 1 轮的 2.63637、第 2 轮的 2.61090858 和第 3 轮的 2.61091327。按验证损失选择第 2 轮（500 步）的检查点。测试答案和 EOS 共计 10,924 个计分 token。

| 400 道冻结测试题 | 预训练 | SFT 第 2 轮 |
| --- | ---: | ---: |
| 答案交叉熵 | 3.4096 | 2.5905 |
| 答案困惑度 | 30.25 | 13.34 |
| 恰好两句完整句子 | 0/400 | 396/400（99.00%） |
| EOS 停止 | 52/400（13.00%） | 399/400（99.75%） |
| 恰好两句且 EOS 停止 | 0/400 | 396/400（99.00%） |

在故事 `100021f2469d…` 中，Jack 看到滑梯后，微调版续写 “He was so excited! He ran over to the slide and started to play.”，保留了开头的情境并结束于第二句。故事 `35adf9e8d12c…` 虽然通过了两句格式检查，主角却从鸭子变成了鱼。故事 `46d11b702141…` 的后半段反复生成颜色词并用完生成额度。微调版还出现了 17 条两句措辞相同的输出（忽略大小写和句末标点）。

## 在 CPU 上运行推理

在仓库根目录运行。`--model sft` 使用训练时的续写指令模板，`--model pretrain` 直接接在英文开头之后生成。两种模式均采用贪心解码，输出新增 token 数、生成耗时、token/s 和停止原因。耗时从解码开始计算，最后生成的 EOS 计入 token 数。用 `--json` 可取得结构化结果。

```powershell
$opening = 'Mia found a paper boat beside the pond. She picked it up and looked for its owner.'
& .\.conda\python.exe generate.py --model sft --prompt $opening
& .\.conda\python.exe generate.py --model pretrain --prompt $opening
```

默认权重来自 `runs/sft/best.pt` 或 `runs/pretrain/best.pt`，分词器来自 `data/prepared/tokenizer.json`。两套本地权重生成后，可以在仓库内重新打包。

```powershell
& .\.conda\python.exe -m scripts.package_weights
```

可将生成的 `dist/tiny-gpt-cpu-weights.zip` 作为 GitHub Release 附件。取得压缩包后，把它放到 `dist` 下并运行以下命令。推理只需仓库源码、本节解压的两套权重和 tokenizer，以及已安装的依赖。

```powershell
Expand-Archive -LiteralPath .\dist\tiny-gpt-cpu-weights.zip -DestinationPath .\dist\weights
$opening = 'Mia found a paper boat beside the pond. She picked it up and looked for its owner.'
& .\.conda\python.exe generate.py --model sft --checkpoint .\dist\weights\sft.pt --tokenizer .\dist\weights\tokenizer.json --prompt $opening
& .\.conda\python.exe generate.py --model pretrain --checkpoint .\dist\weights\pretrain.pt --tokenizer .\dist\weights\tokenizer.json --prompt $opening
```

## 实验产物

公开实验记录包括 [CPU 基准](results/cpu_benchmark.json)、[预训练曲线](results/pretrain_loss_curves.png)、[完整验证指标](results/pretrain_full_validation.json)、[微调测试指标](results/sft_test_metrics.json) 和[同题生成示例](results/sft_test_examples.json)。完整逐题输出由 `train_sft.py --evaluate` 写入 `runs/sft/test_outputs.jsonl`。权重包、训练缓存与可重建的数据位于本地的 Git 忽略目录。

项目用 Python 3.12.16 新建隔离虚拟环境，从 `requirements.txt` 安装依赖后重建了数据，9 个产物的哈希全部一致。小批训练通过，压缩包内的两套权重都完成了 CPU 生成，测速记录保存在 [独立环境推理结果](results/cpu_inference.json)。

## 第二轮优化的固定基线

[基线清单](results/optimization_baseline_v1.json) 记录第一版权重、分词器、数据拆分和逐题结果的 SHA-256，同时保存正式评估设置和按固定种子抽取的 60 个题号。测试题号从 1 开始，`review_order` 是盲评展示顺序。可在仓库根目录用以下命令核验原始产物及题号。

```powershell
& .\.conda\python.exe -m scripts.freeze_baseline
```

正式比较对三套模型使用同一组 400 道题、提示模板、贪心解码、128 token 上下文和 64 token 新增长度上限。答案交叉熵按有效续写 token 和 EOS 汇总，生成端继续统计两句完成率和 EOS 停止率。用验证集选权重。

生成齐三套输出后，按清单顺序展示 60 题。每题用种子 43 的同一个随机数生成器将模型顺序打乱（输入模型名先排序），展示匿名结果并保存单独的对应表，评分后再解盲。先看开头确定人物、相关物件和是否有明确目标，再按以下规则逐条记录。

| 维度 | 记为通过的条件 |
| --- | --- |
| 人物和关键物件承接 | 续写延续已有情境，人物、物件及其角色前后一致。 |
| 行动目标推进 | 续写给出与开头目标直接相关的新行动、线索或结果。开头无明确目标时记为不适用，并报告有效题数。 |
| 无明显重复 | 两句各提供新的信息，句内也没有明显循环用语。 |

每项记为通过、不通过或不适用，并在不通过时记一句理由。纸船题另作演示。

## 构建扩充版训练数据

从仓库根目录执行以下命令。首次使用 `--smoke` 可检查小规模构建，完整数据默认生成到 `data/prepared-v2`。已有输出时，可用 `--output-dir` 指定新的构建目录。构建会复用 `data/raw` 中的分片缓存并核对官方校验值。 [TinyStories 官方分片](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main/data)

```powershell
& .\.conda\python.exe -m scripts.freeze_baseline
& .\.conda\python.exe scripts/prepare_data_v2.py --smoke --output-dir runs/data-v2-smoke
& .\.conda\python.exe scripts/prepare_data_v2.py
```

以种子 42 从四个训练分片各抽取 22,000 条候选，按归一化故事哈希去重并排除官方验证故事。先选新增 SFT 样本，再补充预训练故事。扩充后共有 **80,000 篇预训练故事、17,762,182 个 token**，其中新增 48,000 篇。SFT 训练集共 **8,000 条**，其中新增 4,000 条。预训练验证缓存、400 条 SFT 验证题、400 条测试题和分词器均与第一版逐字节一致。

新增 SFT 样本使用同一故事中紧接开头的两句，要求续写沿用至少一个有内容的词并提供新信息。规则过滤重复句或四词短语、泛化开场及可识别姓名的断接。抽查 12 条可见人物或物件承接，也有与开头行动关系较弱的样本，情节推进会在后续盲评核验。

两次小样本构建逐字节一致。全量检查核对了五个拆分的故事哈希、8,800 个互异的 SFT 开头、80,000 篇预训练故事的 BOS/EOS 边界，以及全部 8,000 条 SFT 的首个答案 token、EOS 和损失掩码。 [扩充版数据清单](results/optimization_data_v2.json)

## 对比两档模型的预训练

A 为 4 层 × 192（2,590,848 参数），B 为 4 层 × 256（4,240,896 参数）。两组均从随机初始化开始，使用同一份扩充数据、4 个 CPU 线程、种子 42、8 × 128 的微批、2 次梯度累积和 0.0003 学习率。先运行 250 步试跑并保存检查点，再续训到累计 12,000 步。两组各处理 24,576,000 个 token。

```powershell
foreach ($experiment in @(@{ Name='a'; Width=192 }, @{ Name='b'; Width=256 })) {
    $phaseArgs = @('--data-dir', 'data/prepared-v2', '--run-dir', "runs/optimization/$($experiment.Name)",
                   '--n-layers', '4', '--d-model', "$($experiment.Width)", '--eval-every', '500')
    & .\.conda\python.exe -u train_pretrain.py @phaseArgs --steps 250
    if ($LASTEXITCODE -ne 0) { throw 'Pilot failed' }
    & .\.conda\python.exe -u train_pretrain.py @phaseArgs --steps 12000 --resume
    if ($LASTEXITCODE -ne 0) { throw 'Pretraining failed' }
}
& .\.conda\python.exe -m scripts.compare_pretraining
```

每组独立存放 `last.pt`、`best.pt`、`metrics.json` 和 `samples.json`。已有运行从对应目录的 `last.pt` 续训，`--steps` 指定累计目标。比较程序核对设置、数据指纹及训练预算，在完整预训练验证缓存上评估两组最佳权重，再汇总到 `results/optimization_pretrain_ab.json`。

两组均完成了 12,000 步。训练期间每次用固定的 32,768 个验证 token 选择最佳权重，随后用完整验证缓存的 323,456 个目标 token 复核。 [指标明细](results/optimization_pretrain_ab.json) · [验证损失曲线](results/optimization_pretrain_ab.png)

| 组别 | 最佳步数 | 完整验证交叉熵 | 完整验证困惑度 | 两段运行耗时 | 峰值工作集 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 12,000 | 2.4849 | 12.00 | 38.1 分钟 | 534 MiB |
| B | 11,500 | 2.3951 | 10.97 | 61.9 分钟 | 597 MiB |

用于第二轮监督微调的预训练权重分别位于 `runs/optimization/a/best.pt` 和 `runs/optimization/b/best.pt`。

## 第二轮微调与盲评

在仓库根目录按相同设置对 A、B 各做三轮全参数 SFT，每轮用冻结的 400 条验证题计算答案交叉熵，以此选择最佳权重。

```powershell
foreach ($group in @('a', 'b')) {
    & .\.conda\python.exe -u train_sft.py --data-dir data/prepared-v2 --pretrained "runs/optimization/$group/best.pt" --run-dir "runs/optimization/$group/sft" --threads 4 --batch-size 16 --seed 42 --epochs 3 --lr 0.0001
    if ($LASTEXITCODE -ne 0) { throw "SFT $group failed" }
}
& .\.conda\python.exe -u -m scripts.evaluate_optimization
```

比较程序生成 400 道题的三方输出 `runs/optimization/comparison/test_outputs.jsonl`，以及 60 道匿名题 `blind_questions.jsonl` 和独立的 `blind_key.json`。前者按原题序记录参考答案、生成 token 和格式判定。测试汇总记录答案交叉熵、两句完成率、EOS 停止率、两句完全重复的次数，以及 4 线程、批大小 16 时的 CPU 批量生成吞吐。

匿名评分保存后运行下面的命令校验匿名题文件指纹和全部评分，再按对应表汇总。

```powershell
& .\.conda\python.exe -m scripts.summarize_blind_review
```

### 第二轮结果与权重选择

A 组选择第 2 轮（1,000 步，验证答案交叉熵 2.1801），B 组选择第 1 轮（500 步，验证答案交叉熵 2.0978）。三套微调模型使用同一份 400 题评估，每套都计分 10,924 个答案 token（含 EOS）。 [测试明细](results/optimization_sft_metrics.json)

| 指标 | 第一版 | A | B |
| --- | ---: | ---: | ---: |
| 答案交叉熵 | 2.5905 | 2.1701 | 2.0877 |
| 恰好两句 | 396/400 | 400/400 | 399/400 |
| EOS 停止 | 399/400 | 400/400 | 400/400 |
| 两句措辞完全重复 | 17 | 13 | 9 |
| 60 题盲评 · 人物和物件承接 | 24/60 | 40/60 | 44/60 |
| 60 题盲评 · 无明显重复 | 38/60 | 29/60 | 38/60 |
| 有明确目标的题 · 推进目标 | 1/6 | 4/6 | 2/6 |

盲评由 Codex 单独完成。评分者先看匿名的 X/Y/Z 输出并保存评分，再用固定对应表汇总。60 题里只有 6 题有明确目标，这一项按有效题数计算。B 的目标推进低于 A，纸船题仍把找船主改成向朋友展示。B 在整批答案交叉熵、完全重复次数以及盲评人物与物件承接上得分最好，因此选为第二轮打包版本。 [匿名评分](results/optimization_blind_scores.json) · [解盲汇总](results/optimization_blind_review.json) · [纸船题和成功、失败样例](results/optimization_inference_examples.json)

| CPU 成本 | 第一版 | A | B |
| --- | ---: | ---: | ---: |
| 参数量 | 2,590,848 | 2,590,848 | 4,240,896 |
| 预训练主体 | 12.6 分钟 | 37.5 分钟 | 61.0 分钟 |
| SFT 历时 | 92.0 秒 | 211.1 秒 | 322.5 秒 |
| 预训练峰值工作集 | 523 MiB | 534 MiB | 597 MiB |
| SFT 峰值工作集 | 661 MiB | 677 MiB | 713 MiB |
| 测试集批量生成吞吐 | 775 token/s | 736 token/s | 470 token/s |

第一版的 SFT 峰值来自同配置复跑，选定权重的张量和逐轮损失均与原始运行一致。批量生成吞吐包含 EOS，使用 4 个 CPU 线程和批大小 16。 [第一版峰值复测](results/optimization_v1_sft_resources.json)

用本地 B 组权重复跑纸船题。命令输入为仓库根目录下的相对路径。

```powershell
$opening = 'Mia found a paper boat beside the pond. She picked it up and looked for its owner.'
& .\.conda\python.exe generate.py --model sft --checkpoint runs/optimization/b/sft/best.pt --tokenizer data/prepared-v2/tokenizer.json --prompt $opening
```

第二轮权重包 `dist/tiny-gpt-cpu-v2-weights.zip` 已通过成员 SHA-256 检查，解压后的预训练与微调权重均完成 CPU 推理。压缩包包含两套权重、分词器、校验清单和数据及训练配置摘要。可用以下命令重新打包。 [训练配置摘要](results/optimization_release_summary.json)

```powershell
& .\.conda\python.exe -m scripts.package_weights --pretrain runs/optimization/b/best.pt --sft runs/optimization/b/sft/best.pt --tokenizer data/prepared-v2/tokenizer.json --summary results/optimization_release_summary.json --output dist/tiny-gpt-cpu-v2-weights.zip
```
