import json
import time
import threading
import websocket
import requests
from python.capture.rank_crawler import load_config, get_access_token

# WebSocket OpCodes
OP_DISPATCH = 0
OP_HEARTBEAT = 1
OP_IDENTIFY = 2
OP_HELLO = 10
OP_HEARTBEAT_ACK = 11

class BotClient:
    def __init__(self, app_id, token, sandbox=True):
        self.app_id = app_id
        self.token = token # 这里是 AccessToken
        self.sandbox = sandbox
        self.ws = None
        self.heartbeat_interval = 40
        self.last_seq = None
        self.api_base = "https://sandbox.api.sgroup.qq.com" if sandbox else "https://api.sgroup.qq.com"

    def get_gateway(self):
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
        t = data.get("t") # Event Type
        d = data.get("d") # Event Data
        s = data.get("s") # Sequence

        if s:
            self.last_seq = s

        if op == OP_HELLO:
            self.heartbeat_interval = d["heartbeat_interval"] / 1000
            print(f"[WS] Hello received. Heartbeat interval: {self.heartbeat_interval}s")
            # 启动心跳线程
            threading.Thread(target=self.send_heartbeat, daemon=True).start()
            # 发送鉴权
            self.send_identify()

        elif op == OP_DISPATCH:
            print(f"[Event] {t}")
            if t == "GROUP_AT_MESSAGE_CREATE":
                group_openid = d.get("group_openid")
                content = d.get("content", "").strip()
                print("\n" + "="*50)
                print(f"!!! 捕获到群消息 !!!")
                print(f"Group OpenID: {group_openid}")
                print(f"Content: {content}")
                print("="*50 + "\n")
                print("您可以复制上面的 Group OpenID 填入 config.yaml")
                ws.close() # 拿到 ID 后直接退出

    def send_heartbeat(self):
        while self.ws and self.ws.keep_running:
            payload = {
                "op": OP_HEARTBEAT,
                "d": self.last_seq
            }
            try:
                self.ws.send(json.dumps(payload))
                # print("[WS] Heartbeat sent")
            except:
                break
            time.sleep(self.heartbeat_interval)

    def send_identify(self):
        payload = {
            "op": OP_IDENTIFY,
            "d": {
                "token": f"QQBot {self.token}",
                "intents": 1 << 25, # GROUP_AT_MESSAGES (需要确保后台已开启此 Intent)
                "shard": [0, 1]
            }
        }
        print("[WS] Sending Identify...")
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
    print(">>> QQBot Group OpenID 获取工具")
    config = load_config('config.yaml')
    bot_config = config.get('qq_bot', {})
    
    app_id = bot_config.get('app_id')
    client_secret = bot_config.get('client_secret')
    sandbox = bot_config.get('sandbox', True)
    
    if not app_id or not client_secret:
        print("错误: 请先在 config.yaml 中填入 app_id 和 client_secret")
        return

    print("1. 获取 AccessToken...")
    token = get_access_token(app_id, client_secret)
    if not token:
        print("获取 Token 失败")
        return
        
    print("2. 连接 WebSocket 监听群消息...")
    print("请现在去目标群里，@机器人 并随便发一条消息 (例如: @Bot hello)")
    
    client = BotClient(app_id, token, sandbox)
    client.run()

if __name__ == '__main__':
    main()