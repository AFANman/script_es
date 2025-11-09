## 1. 实施清单（Python 版）
- [x] 1.1 创建规格增量 `changes/add-adb-scroll-capture/specs/adb-scroll-capture/spec.md`
- [x] 1.2 模块化重构：`python/capture/` 与 `python/parse/`，入口均为 `main.py`
- [x] 1.3 实现 ADB 客户端与编排器（截图/滚动/翻页/结束判定）
- [x] 1.4 README 简化为“三步使用”，`openspec/project.md` 更新项目说明
- [x] 1.5 `.gitignore` 忽略非代码产物（截图、模型权重、缓存等）
- [x] 1.6 解析模块默认命令输出 `activity_items.json`

## 2. 验证
- [x] 2.1 在示例设备上采集多页截图（如 7 页），命名递增且无误
- [x] 2.2 解析输出包含 `items` 与 `missing`，与预期一致
- [x] 2.3 手动检查末页补滑与底部空白检测有效

## 3. 存档
- [x] 3.1 将本变更集移动至 `openspec/changes/archive/`