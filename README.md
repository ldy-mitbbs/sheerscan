# sheerscan

扫描一段视频，**找出出现薄透丝袜（锦纶/尼龙袜）的时间点**。

为 **高召回 + 语义过滤 + 人工复核** 而设计 —— 而不是去追求某个"置信度分数"。一个关键经验：视觉大模型给出的 high/medium/low 是临时编造的 token，在这个任务上甚至与真值**反相关**（真实的浅淡场景常返回 `low`，光滑裸腿的误判反而返回 `high`），所以本 pipeline 不用置信度门控。

> 从一个更大的媒体管理应用里抽出来，既能独立使用，也能作为依赖被宿主应用嵌入。

---

## 它是什么 / 不是什么

- **是**：一个把"2 小时视频里几百个噪声候选 → 二十几个干净、真实的场景"收敛下来的工具链，最后交给人快速复核。
- **不是**：一个一键给出"有/无"的全自动分类器。最难的那类误判（光滑裸腿被读成肉色丝袜）在单帧层面**无法**靠后处理或更强的模型分开 —— 产品答案就是高召回 + 语义过滤 + 人工复核。

---

## 核心设计取舍（为什么长这样）

这些是反复调参后沉淀下来的结论，**没有新证据不要轻易推翻**：

1. **只做粗筛（coarse-only）**。加上"精筛视频复核"那一道**同时拉低了召回和精度**（它只处理一小部分候选，还确认错了对象）。单独粗筛召回好得多。
2. **简单粗筛 prompt**。老的 ~600 字嵌套规则 prompt 容易"刷屏"误报；简单 prompt 直接点名明显的非目标。
3. **不做置信度门控**。VLM 的 high/medium/low 不是校准过的统计量，在本类任务上反相关（见上）。所以粗筛 prompt 干脆不再要求给置信度，唯一判断就是二元的"有没有丝袜"。
4. **二次机会裁剪召回**（`INSPECTOR_SECOND_CHANCE=1`，默认开）。粗筛稳定漏掉"腿脚区域只占画面一小块"的场景——全帧提分辨率救不回来（实测纯波动），姿态定位的原生分辨率裁剪可以。对粗筛没报的采样帧：先在已抽出的帧上本地跑姿态检测（零解码、零 API），找到腿部的才解码原生帧、裁剪腿部、再问同一个粗筛模型一次。实测恢复 ~46% 的漏检事件，干净负样本区域只多 ~8% 误火。幸存者并入粗筛候选池，走同样的去重+语义过滤。`INSPECTOR_SECOND_CHANCE_MAX_CROPS`（默认 400）封顶成本。需要 `[cropzoom]` extra；姿态 worker 不可用时自动跳过（fail-open 回到纯粗筛）。
5. **语义 reason 过滤是唯一裁判**（`reason_filter.py`）。一个**本地**小模型（Ollama，默认 `qwen2.5:3b`）逐条读候选的中文 `reason` 描述，只丢掉它判定为明确"无丝袜"的（长裤盖住、明显凉鞋裸脚、清楚裸足、广告图/家具等），保留"有"和"不确定"。这取代了脆弱的关键词匹配 —— 让模型读懂**语义**，比关键词和置信度都稳。Ollama 不可用时**fail-open**（全保留，绝不静默丢弃）。

> 实测一个 2 小时视频：~184 个噪声候选 → ~22 个干净、真实的场景。

**速度**：抽帧用单次 ffmpeg 关键帧 pass（`-skip_frame nokey`），比逐帧 PyAV 解码快约 30×；粗筛批次并发跑。一个 ~2 小时视频从约 25 分钟降到约 1.4 分钟。

---

## 安装

```bash
pip install git+https://github.com/ldy-mitbbs/sheerscan.git
# 或克隆后可编辑安装
git clone https://github.com/ldy-mitbbs/sheerscan.git && cd sheerscan
pip install -e .
```

**系统依赖**：

- **ffmpeg**（命令行）—— 抽帧主路径直接调用 `ffmpeg`，必须在 `PATH` 里。
  - macOS：`brew install ffmpeg`；Debian/Ubuntu：`apt install ffmpeg`
- **Ollama**（可选但推荐）—— 语义 reason 过滤用它跑本地小模型。没有它则过滤层 fail-open（保留全部候选）。
  - 装好后：`ollama pull qwen2.5:3b`

**可选 extras**：

```bash
pip install "sheerscan[web]"       # 挂载 Flask 复核界面（/inspect、/corpus）
pip install "sheerscan[cropzoom]"  # 裁剪放大复核（ultralytics 姿态定位腿/脚）
pip install "sheerscan[clip]"      # 本地 CLIP 预筛（默认关）
pip install "sheerscan[test]"      # 跑测试
```

**视觉大模型 API**：粗筛走云端 VLM。默认 provider 是 `mulerouter`，也支持 `openrouter`。把 key 放进环境变量或配置文件（见下）：

```bash
export MULEROUTER_API_KEY=...      # 或 OPENROUTER_API_KEY
```

---

## 快速开始

### 命令行

```bash
# 识别单个视频，打印命中的时间点
sheerscan inspect /path/to/video.ts --interval 5

# 回归 harness（见下文）
sheerscan replay --mode postprocess
sheerscan corpus stats
```

### Web 复核界面

```python
from flask import Flask
from sheerscan.inspector import VideoInspectorJobManager
from sheerscan.web import create_inspector_blueprint

app = Flask(__name__)
app.register_blueprint(create_inspector_blueprint(
    job_manager=VideoInspectorJobManager(),
    cache=None,          # 可传入自带的缓存（需有 get_llm/put_llm）
))
app.run(port=8765)
# 浏览器打开 /inspect（任务 + 检测复核）、/corpus（语料标注）
```

---

## 配置

所有旋钮通过环境变量或 `~/.config/sheerscan/config.json` 读取（环境变量优先）。常用项：

| Key | 默认 | 说明 |
|---|---|---|
| `INSPECTOR_API_PROVIDER` | `mulerouter` | VLM provider（`mulerouter` / `openrouter`）|
| `MULEROUTER_API_KEY` / `OPENROUTER_API_KEY` | — | API key（建议放环境变量）|
| `INSPECTOR_MODEL` | `qwen3.6-flash` | 粗筛模型 |
| `INSPECTOR_INTERVAL` | `5` | 抽帧间隔（秒），越小候选越多 |
| `INSPECTOR_REASON_FILTER` | `0` | 开启语义 reason 过滤（强烈建议开）|
| `INSPECTOR_REASON_FILTER_MODEL` | `qwen2.5:3b` | reason 过滤用的本地 Ollama 模型 |
| `GPU_BASE_URL` | `localhost` | Ollama 主机（自动补 `:11434`）|
| `INSPECTOR_COARSE_CONCURRENCY` | `6` | 粗筛批次并发数 |
| `INSPECTOR_CROP_ZOOM` | `0` | 开启裁剪放大复核（需 `[cropzoom]`）|
| `INSPECTOR_SECOND_CHANCE` | `1` | 二次机会裁剪召回（需 `[cropzoom]`；姿态 worker 不可用时自动跳过）|
| `INSPECTOR_SECOND_CHANCE_MAX_CROPS` | `400` | 单视频二次机会姿态扫描帧数上限 |
| `SHEERSCAN_LOCAL_DIR` | `~/sheerscan-local` | 任务产物 / 帧 / 语料的本地目录 |

> ⚠️ 别开 `INSPECTOR_EXTREME_RECALL` 和 `INSPECTOR_STRICT_EVIDENCE_FILTER` —— 前者是隐藏的刷屏来源，后者会严重砸召回。

> 📦 `inspector.py` 里还保留着一批**默认关闭的可选/实验功能**（精筛视频 pass、native_video / screenshot 模式、verifier 复核、本地 CLIP 预筛、crop-zoom），都靠对应 flag 挡着、不在默认 pipeline 里跑——是调参历史和 A/B 留下的 feature flag，不是死代码。文件顶部的导览注释标了哪些是核心、哪些是 opt-in。独立安装默认就是推荐的「粗筛 + 语义过滤」那套。

---

## 作为库嵌入

默认独立运行：配置读环境变量 / `~/.config/sheerscan/config.json`，路径按本地处理。

宿主应用可以注入自己的配置存储和"容器↔主机"路径映射，**而不改动 pipeline 里任何调用点**：

```python
import sheerscan

class MySettings:
    def get_setting(self, name, default=None): ...
    def get_secret(self, name): ...
    def get_local_dir(self): ...     # -> pathlib.Path

class MyPathMapper:
    def to_host(self, path): ...      # 容器/远端看到的路径 -> 本机路径
    def to_container(self, path): ...

sheerscan.configure(settings=MySettings(), pathmap=MyPathMapper())
```

> 注意：如果视频路径是容器/远端视角的（如 `/data/...`），**必须**注入 `pathmap`，否则默认按本地路径找文件会失败。

---

## 调参与回归 harness

改 prompt / 模型 / 过滤 / 采样后，用回归 harness 判断质量是否真的变好，别靠单次肉眼。

- **语料（ground truth）**：`sheerscan/corpus.py`。语料 manifest（`tests/inspector_corpus/manifest.jsonl`，含哈希 `video_id` + 中文 `reason` 描述）、真实帧、`video_id→路径` 映射**都留在本地、不随仓库发布**（gitignored）。仓库只保留 `baseline.json` 作为回归目标。本地跑 `sheerscan corpus harvest` 会在本机生成 manifest。
- **重放对比**：
  - `sheerscan replay --mode postprocess` —— **免费、确定性** 的重放：把每个已录 trace 通过*当前*后处理代码重建检测并对语料打分。改过滤/去重/窗口逻辑时用它。`--set key=value` 覆盖某个旋钮，`--compare baseline.json` 看 recall/precision 增量。重放会跑**完整的现役 pipeline**：语义 reason 过滤（`INSPECTOR_REASON_FILTER` 开时；判定走本地 Ollama，缓存命中后免费）和 crop-zoom 闸（`--set crop_gate=0.3`，复用 live run 存在 result.json 里的 `crop_score`）。语料里**没有任何 trace** 的视频会列在 `uncovered_videos` 里、不参与打分——那是 harness 覆盖率问题，不是 pipeline 质量问题。
  - ⚠️ 实测**别开 crop gate**（`INSPECTOR_CROP_ZOOM_DROP_BELOW`）：现用的 rescore prompt+model 过于保守，真阳的 crop_score 中位数也只有 ~0.1，任何阈值都是 1:1 拿召回换精度。先换/校准 rescore 模型再谈闸。
  - `sheerscan replay --mode frames --model <id>` —— 在已标注帧上做模型/prompt 的 A/B（**会花 API**）。
- **诚实精度需要负样本**：一个未匹配的检测，只有落在标注的负样本附近才算误报，否则算"未知"。
- **回归测试**：`tests/test_inspector_regression.py` —— 一个可移植 fixture 测试 + 一个真实数据测试（断言 recall/precision ≥ `baseline.json`，无本地 trace 时自动跳过）。

---

## 已知局限

最主要的一类误报 —— **光滑裸腿被读成肉色丝袜** —— 在单帧层面**无法**靠任何后处理或换模型分开：真阳和假阳用的措辞/置信度完全一样，更强的模型、视频/运动信息、更严的 prompt 都不能把它们分开（测过 5 种模型/prompt/模态组合，全是 1:1 拿召回换精度）。所以产品答案是高召回 + 语义过滤 + **人工复核**（`/inspect` 的 Quick Review）。

---

## 目录结构

```
sheerscan/
  inspector.py        核心 pipeline + 任务管理（VideoInspector / VideoInspectorJobManager）
  reason_filter.py    本地小模型语义过滤（唯一裁判）
  crop_zoom.py        裁剪放大复核（可选）
  corpus.py           回归语料 ground truth
  replay.py           回归 harness（免费重放 / 模型 A/B）
  runtime.py          配置 + 路径映射注入层（独立默认 + 可被宿主覆盖）
  ollama.py           最小 Ollama 客户端（仅 stdlib）
  cache.py            可选的判定缓存
  web/                可挂载的 Flask blueprint + 单文件前端
  cli.py              独立 CLI
tests/                回归测试 + 语料 manifest/baseline
```

---

## 许可证

MIT，见 [LICENSE](LICENSE)。

## 说明

本项目针对的是视频中"薄透丝袜"这一视觉属性的检测，语料里的 `reason` 是中文场景描述。仅含视觉外观判断，不含任何可识别版权作品的元数据（文件名、发布组、tracker 等均未提交）。
