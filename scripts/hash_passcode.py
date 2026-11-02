"""生成员工口令的 SHA-256 哈希（用于 configs/default.json 的 staff.passcode_sha256）。

用法：
    python scripts/hash_passcode.py 我的口令
输出可直接粘贴到配置的 passcode_sha256 字段；粘贴后删除明文 passcode 字段更安全。
"""
from __future__ import annotations

import getpass
import hashlib
import sys


def main() -> None:
    if len(sys.argv) > 1:
        passcode = sys.argv[1]
    else:
        passcode = getpass.getpass("请输入口令（输入不回显）：")
    if not passcode.strip() or len(passcode) < 6:
        print("口令至少 6 位，请重试", file=sys.stderr)
        raise SystemExit(1)
    digest = hashlib.sha256(passcode.encode("utf-8")).hexdigest()
    print("passcode_sha256:")
    print(digest)
    print("\n使用方法：把上面的值填入 configs/default.json 的 staff.passcode_sha256，")
    print("并删除 staff.passcode 明文字段。员工登录时仍输入原口令。")


if __name__ == "__main__":
    main()
