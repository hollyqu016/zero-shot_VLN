"""
跨阶段共用的小工具。

放这里的东西必须满足：被两个以上阶段用到，且不依赖 agent 状态。
"""

import json
import re
import socket
import os


def _setup_proxy(preferred_url: str = "http://127.0.0.1:7897") -> None:
    """Set proxy only when:
    1. User has NOT already set http_proxy/https_proxy env vars
    2. The preferred proxy is TCP-reachable (otherwise skip)
    """
    if os.environ.get("http_proxy") or os.environ.get("https_proxy"):
        return
    try:
        host, port = preferred_url.replace("http://", "").replace("/", "").split(":")
        with socket.create_connection((host, int(port)), timeout=1.0):
            os.environ["http_proxy"] = preferred_url
            os.environ["https_proxy"] = preferred_url
    except Exception:
        pass

def robust_json_parse(json_str: str):
    """
    Parse a JSON string that may contain comments or common formatting errors; returns dict or None.
    """

    json_str = re.sub(r'//.*', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

    json_str = json_str.strip()

    json_str = re.sub(r'^```json', '', json_str)
    json_str = re.sub(r'^```', '', json_str)
    json_str = re.sub(r'```$', '', json_str)

    try:
        return json.loads(json_str)
    except Exception as e:
        pass
        return None
