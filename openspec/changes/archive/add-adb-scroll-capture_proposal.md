# 变更提案：ADB 滚动采集（已完成并存档）

## 摘要
实现一个“开箱即用”的安卓分页滚动采集器，通过 ADB 自动执行：截图 → 滚动 → 翻页，直到末页或达到上限。并将项目结构调整为两个模块：`capture/` 与 `parse/`，各自使用 `main.py` 作为启动器。README 简化为“介绍 + 三步使用”，`.gitignore` 忽略非代码产物。

## 交付范围
- Python 采集模块：`python/capture/`
  - `main.py`（启动器）：支持最小命令运行。
  - `orchestrator.py`：编排截图、滚动、翻页与结束判定。
  - `adb_client.py`：封装设备连接、截图、滑动、点击、按键事件。
- Python 解析模块：`python/parse/`
  - `main.py`：默认解析 `screenshots/` 为 `activity_items.json`。
  - `activity_parser.py`：行对齐、分数识别、噪声过滤与缺失段统计。
- 文档与配置：
  - README 简化为“项目简介 + 使用方法（3 步）”。
  - `openspec/project.md` 更新为中文项目说明（模块结构、解析逻辑）。
  - `.gitignore` 忽略虚拟环境、缓存、模型权重、截图与输出等非代码内容。

## 默认行为与命令
- 最小运行：
  - 采集：`python python/capture/main.py`
  - 解析：`python python/parse/main.py`
- 可选覆盖：
  - 指定设备：`--device-id 192.168.0.101:5555`
  - 滚动模式：`--mode tap` 或 `--mode keyevent`
- 输出：截图保存至 `screenshots/`；解析输出 `activity_items.json`。

## 结束判定与稳健性
- 翻到末页时进行一次补滑，避免遗漏底部内容。
- 达到截图计数上限或不再出现新内容时结束采集。
- 自动选择在线设备，若无则尝试 `127.0.0.1:5555`。

## 验收与结果
- 在示例设备上成功采集多页截图（如 7 页），文件命名按页与序号递增。
- 解析产出包含 `items`（分数-描述对）与 `missing`（缺失分数段）。

## 状态
- 已完成，实现与文档更新到位，并移入 `openspec/changes/archive/` 存档。