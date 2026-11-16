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
