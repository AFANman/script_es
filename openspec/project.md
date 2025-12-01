# 项目说明（Project Context）

## 目的（Purpose）
- 提供一个“开箱即用”的安卓分页滚动截图工具，通过 ADB 自动执行“截图 → 滚动 → 翻页”直到末页。
- 提供解析模块，将截图中的活动道具与分数识别为结构化 JSON 结果，包含缺失分数段统计。

## 技术栈（Tech Stack）
- Python：`easyocr`（OCR）、`opencv-python`（图像处理）、`numpy`。
- ADB：设备连接、截图、滑动、点击、按键事件。
- Node 版本仍保留于 `src/`（不再作为推荐入口）。

## 模块与结构（Architecture）
- 采集模块：`python/capture/`
  - 三层架构（Tidy First）：
    - `adb_client.py`：基础 ADB 封装（设备连接、截图、滑动、点击、按键）。
    - `abd_command.py`：基于 `adb_client` 的原子业务操作封装（`ensure_dir`、`wait`、`tap`、`capture`）。
    - `activity_flows.py`：编排 `abd_command` 的原子操作以达成业务流程（积分线、奖杯分数线）。
  - 启动入口：
    - `flows_main.py`：活动积分/奖杯分数线执行模板（调用 `activity_flows`，输出到 `captures/`）。
    - `main.py`：分页滚动采集启动器（调用 `orchestrator.py`，支持自动滑动与翻页）。
  - `orchestrator.py`：通用截图/滚动/翻页编排与默认参数计算（自动滑动、自动翻页坐标、末页补滑）。
- 解析模块：`python/parse/`
  - `main.py`：解析启动器（默认读取 `screenshots` 输出 `activity_items.json`）。
  - `activity_parser.py`：解析核心逻辑（行对齐、分数识别、噪声过滤）。

## 快速使用（Quick Start）
- 安装依赖：`python -m pip install -r python/requirements.txt`
- 活动脚本（积分线/奖杯分数线）：`python -m python.capture.flows_main`
- 分页滚动采集：`python python/capture/main.py`
- 解析：`python python/parse/main.py`

可选覆盖：
- 指定设备：`python python/capture/main.py --device-id 192.168.0.101:5555`
- 点击滚动：`python python/capture/main.py --mode tap`
 - 活动脚本输出根目录：`python -m python.capture.flows_main --out-root captures`
 - 活动脚本等待间隔：`python -m python.capture.flows_main --wait-seconds 1.0`
- 按键滚动：`python python/capture/main.py --mode keyevent`

默认行为（Defaults）
- 自动选择在线设备或连接 `127.0.0.1:5555`。
- 自动计算稳定的上滑参数（中轴从下向上）与翻页坐标（底部右/左）。
- 截图默认保存到 `screenshots/`，解析默认输出 `activity_items.json`。

## 解析逻辑概要（Parsing Logic）
- 使用 EasyOCR 提取文本及 bbox，支持可选 ROI，默认小 batch 和单线程。
- 在右半区域识别分数（支持千分位），仅保留 1000 的倍数字段为有效分数。
- 以分数的垂直中心为参考，聚合其左侧同行文字，按 x 排序拼接为“道具描述”。
- 规范化数量表达（统一乘号为 `x`，如 `PAIRINGx1` → `PAIRING x1`），过滤噪声（比值 `1/7`、纯数字、符号、短大写等 UI 干扰）。
- 对同一分数保留更完整描述（优先长文本），输出 `items` 列表与 `missing`（从 1000 到 `--max` 的缺失分数段）。

## 约定（Conventions）
- 启动器文件统一命名为 `main.py`，模块按功能分目录（`capture/`、`parse/`）。
- 文档：README 只保留“项目简介 + 使用方法”；进阶说明见 `AGENTS.md`。

## 测试（Testing）
- Node 端保留 `tests/orchestrator.test.js`；Python 暂不含自动测试。
- 手动验证建议：运行采集后检查 `screenshots/`；解析后检查 `activity_items.json` 的 `items` 与 `missing` 字段。

## 约束（Constraints）
- 需要本地 ADB 可用并能连接目标设备或模拟器。
- OCR 识别效果与页面设计、清晰度相关；极端 UI 可能需手工配置 ROI 或模式。

## 外部依赖（External Dependencies）
- ADB（Android Platform Tools）。
- Python 包：`easyocr`、`opencv-python`、`numpy`。

## 参考（References）
- 用法与示例：见 `README.md`。
- 进阶参数与 Node 用法：见 `AGENTS.md`。

## 当前输出（Outputs）
- 活动脚本：
  - `captures/point_line/point_line_01.png` ～ `point_line_04.png`
  - `captures/trophy_line/trophy_line_01.png` ～ `trophy_line_04.png`
- 分页滚动采集：
  - 默认输出到 `capture/`（可用 `--prefix` 覆盖）
