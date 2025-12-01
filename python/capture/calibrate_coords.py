import easyocr
import os
from python.capture.adb_client import AdbClient

def main():
    client = AdbClient()
    client.connect()
    
    print("正在截图以校准坐标...")
    img_path = "calibration.png"
    client.screencap_to_file(img_path)
    
    print("正在识别文字...")
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    results = reader.readtext(img_path)
    
    targets = ["1位", "100位", "1,000位", "5,000位", "10,000位", "50,000位"]
    # 注意 OCR 可能识别为 "100位~" 或 "100位"
    
    print("\n识别结果 (目标关键字):")
    found = {}
    
    for bbox, text, conf in results:
        t = text.replace('~', '').strip()
        # 简单的模糊匹配
        for target in targets:
            clean_target = target.replace(',', '')
            clean_text = t.replace(',', '')
            if clean_target in clean_text:
                # 计算中心点
                x_center = int((bbox[0][0] + bbox[1][0]) / 2)
                y_center = int((bbox[0][1] + bbox[2][1]) / 2)
                print(f"找到 '{text}': ({x_center}, {y_center})")
                found[target] = (x_center, y_center)
                
    print("\n建议更新 abd_command.py 中的坐标:")
    for t in targets:
        if t in found:
            print(f"{t}: {found[t]}")
        else:
            print(f"{t}: 未找到")

if __name__ == '__main__':
    main()
