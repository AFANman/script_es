import argparse
from activity_parser import parse_directory, save_json, parse_image_items
import easyocr
import os


def main():
    parser = argparse.ArgumentParser(description='解析活动截图为道具-分数 JSON')
    parser.add_argument('--dir', default='screenshots', help='截图目录（默认 screenshots）')
    parser.add_argument('--file', help='单张截图路径（设置后仅解析该图片）')
    parser.add_argument('--out', default='activity_items.json', help='输出 JSON 路径')
    parser.add_argument('--max', type=int, default=100000000, help='完整性校验最大分数，默认 100000000')
    parser.add_argument('--models', default='python/.easyocr', help='模型缓存目录（默认 python/.easyocr）')
    parser.add_argument('--download', action='store_true', help='允许首次运行自动下载模型')
    parser.add_argument('--workers', type=int, default=0, help='OCR 识别 DataLoader workers，默认 0')
    parser.add_argument('--batch', type=int, default=4, help='OCR 识别 batch_size，默认 4')
    parser.add_argument('--roi', nargs=4, type=float, help='裁剪区域 [left top right bottom]；比例 0-1 或像素')
    parser.add_argument('--debug', action='store_true', help='启用调试输出（打印识别 token）')
    args = parser.parse_args()

    print('初始化 OCR（easyocr）...')
    # 确保模型目录存在；若缺失检测模型且未显式允许下载，则自动开启下载
    os.makedirs(args.models, exist_ok=True)
    detector_pth = os.path.join(args.models, 'craft_mlt_25k.pth')
    allow_download = bool(args.download)
    if not os.path.exists(detector_pth) and not allow_download:
        print(f'未找到检测模型 {detector_pth}，自动启用下载。')
        allow_download = True

    # 初始化 OCR （中文简体 + 英文），设置本地模型目录与下载开关
    reader = easyocr.Reader(
        ['ch_sim', 'en'],
        gpu=False,
        model_storage_directory=args.models,
        download_enabled=allow_download,
        verbose=True,
    )

    read_opts = {'batch_size': args.batch, 'workers': args.workers}
    if args.roi:
        read_opts['roi'] = args.roi
        print('使用 ROI 裁剪：', args.roi)
    if args.debug:
        read_opts['debug'] = True

    # 单图解析分支
    if args.file:
        print(f'开始解析单图：{args.file} ...')
        pairs = parse_image_items(reader, args.file, read_options=read_opts)
        # 构造与目录解析一致的输出结构
        score_map = {}
        for p in pairs:
            s = p['score']
            if s not in score_map or len(p['item']) > len(score_map[s]):
                score_map[s] = p['item']
        items = [{'score': s, 'item': score_map[s]} for s in sorted(score_map.keys())]
        missing = [s for s in range(1000, args.max + 1, 1000) if s not in score_map]
        res = {'items': items, 'missing': missing}
        save_json(args.out, res)
        print(f'解析完成（单图）：共 {len(items)} 条；缺失 {len(missing)} 个分数段。')
        if items:
            print('识别结果示例：', items[:5])
        if missing:
            print('缺失分数段示例（最多20）：', ', '.join(map(str, missing[:20])))
    else:
        print(f'开始解析目录：{args.dir} ...')
        res = parse_directory(reader, args.dir, max_score=args.max, read_options=read_opts)
        save_json(args.out, res)
        print(f'解析完成：共 {len(res["items"])} 条；缺失 {len(res["missing"]) } 个分数段。')
        if res['missing']:
            print('缺失分数段示例（最多20）：', ', '.join(map(str, res['missing'][:20])))


if __name__ == '__main__':
    main()