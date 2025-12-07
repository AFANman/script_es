import json
import time
import threading
import websocket
import requests
import re
import os
from python.capture.rank_crawler import load_config, get_access_token

# WebSocket OpCodes
OP_DISPATCH = 0
OP_HEARTBEAT = 1
OP_IDENTIFY = 2
OP_HELLO = 10
OP_HEARTBEAT_ACK = 11

class BotClient:
    def __init__(self, app_id, client_secret, sandbox=True):
        self.app_id = app_id
        self.client_secret = client_secret
        self.sandbox = sandbox
        self.ws = None
        self.token = None
        self.heartbeat_interval = 40
        self.last_seq = None
        self.api_base = "https://sandbox.api.sgroup.qq.com" if sandbox else "https://api.sgroup.qq.com"
        
    def refresh_token(self):
        self.token = get_access_token(self.app_id, self.client_secret)
        return self.token

    def get_gateway(self):
        if not self.token:
            self.refresh_token()
        url = f"{self.api_base}/gateway/bot"
        headers = {"Authorization": f"QQBot {self.token}"}
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            return resp.json().get("url")
        print(f"获取 Gateway 失败: {resp.text}")
        return None

    def on_message(self, ws, message):
        data = json.loads(message)
        op = data.get("op")
        t = data.get("t")
        d = data.get("d")
        s = data.get("s")

        if s:
            self.last_seq = s

        if op == OP_HELLO:
            self.heartbeat_interval = d["heartbeat_interval"] / 1000
            threading.Thread(target=self.send_heartbeat, daemon=True).start()
            self.send_identify()

        elif op == OP_DISPATCH:
            if t == "GROUP_AT_MESSAGE_CREATE":
                self.handle_group_message(d)

    def handle_group_message(self, data):
        content = data.get("content", "").strip()
        group_openid = data.get("group_openid")
        msg_id = data.get("id")
        
        print(f"[Bot] 收到群消息: {content} (Group: {group_openid})")
        
        # 简单的指令匹配
        if "排名" in content or "rank" in content.lower():
            self.reply_latest_report(group_openid, msg_id)

    def reply_latest_report(self, group_openid, msg_id):
        print(f"[Bot] 收到查询请求，准备读取最新报告 (Ref MsgID: {msg_id})...")
        
        report_path = 'rank_data/result.txt'
        if not os.path.exists(report_path):
            self.reply_message(group_openid, msg_id, "暂无排名数据，请稍后再试。")
            return
            
        try:
            # 获取文件修改时间
            mtime = os.path.getmtime(report_path)
            time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
            
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
                
            # 拼接更新时间
            final_msg = f"【更新时间: {time_str}】\n{report_content}"
            
            self.reply_message(group_openid, msg_id, final_msg)
                
        except Exception as e:
            print(f"[Bot] 读取报告失败: {e}")
            self.reply_message(group_openid, msg_id, "读取数据失败。")

    def reply_message(self, group_openid, msg_id, content):
        if not self.token:
            self.refresh_token()
            
        url = f"{self.api_base}/v2/groups/{group_openid}/messages"
        headers = {
            "Authorization": f"QQBot {self.token}",
            "Content-Type": "application/json"
        }
        payload = {
            "content": content,
            "msg_type": 0,
            "msg_id": msg_id 
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code in [200, 202]:
                print(f"[Bot] 回复成功")
            else:
                print(f"[Bot] 回复失败: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"[Bot] 回复异常: {e}")

    def send_heartbeat(self):
        while self.ws and self.ws.keep_running:
            payload = {
                "op": OP_HEARTBEAT,
                "d": self.last_seq
            }
            try:
                self.ws.send(json.dumps(payload))
            except:
                break
            time.sleep(self.heartbeat_interval)

    def send_identify(self):
        payload = {
            "op": OP_IDENTIFY,
            "d": {
                "token": f"QQBot {self.token}",
                "intents": 1 << 25, # GROUP_AT_MESSAGES
                "shard": [0, 1]
            }
        }
        self.ws.send(json.dumps(payload))

    def run(self):
        gateway_url = self.get_gateway()
        if not gateway_url:
            return
        
        print(f"[WS] Connecting to {gateway_url}...")
        self.ws = websocket.WebSocketApp(
            gateway_url,
            on_message=self.on_message,
            on_error=lambda ws, err: print(f"[WS Error] {err}"),
            on_close=lambda ws, *args: print("[WS] Closed")
        )
        self.ws.run_forever()

def main():
    print(">>> 启动 Bot 监听服务...")
    config = load_config('config.yaml')
    bot_config = config.get('qq_bot', {})
    
    app_id = bot_config.get('app_id')
    client_secret = bot_config.get('client_secret')
    sandbox = bot_config.get('sandbox', True)
    
    if not app_id or not client_secret:
        print("错误: config.yaml 中未配置 qq_bot (app_id, client_secret)")
        return

    client = BotClient(app_id, client_secret, sandbox)
    client.run()

if __name__ == '__main__':
    main()