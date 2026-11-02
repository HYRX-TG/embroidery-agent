"""把 Markdown 手册渲染成自包含 HTML（离线可看），供浏览器打开。

用法：
    python scripts/render_docs.py                # 渲染 docs/使用手册.md → docs/使用手册.html
    python scripts/render_docs.py --file xxx.md  # 指定文件
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from markdown_it import MarkdownIt
except ImportError:
    print("需要安装 markdown-it-py：pip install markdown-it-py", file=sys.stderr)
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parents[1]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>
:root {{
  --ink: #2b2118; --ink-soft: #4a3b2c; --gold: #a8865a; --gold-light: #d9c6a3;
  --gold-pale: #f3e9d6; --bg: #f7f2e9; --card: #fff; --muted: #8a7a66;
  --code-bg: #f3ede1; --border: #e6dcc9;
}}
* {{ box-sizing: border-box; }}
body {{
  font-family: "PingFang SC", "Microsoft YaHei", "Noto Serif SC", Georgia, serif;
  background: var(--bg); color: var(--ink); margin: 0;
  line-height: 1.75; font-size: 15.5px;
}}
.wrap {{ max-width: 880px; margin: 0 auto; padding: 40px 32px 80px; }}
.doc {{
  background: var(--card); border: 1px solid var(--border); border-radius: 16px;
  padding: 44px 52px; box-shadow: 0 4px 24px rgba(43,33,24,0.07);
}}
.doc h1 {{
  font-size: 28px; color: var(--ink); border-bottom: 2px solid var(--gold);
  padding-bottom: 14px; margin: 0 0 24px; letter-spacing: 1px;
}}
.doc h2 {{
  font-size: 21px; color: var(--ink); margin: 36px 0 14px;
  padding-left: 12px; border-left: 4px solid var(--gold); letter-spacing: 0.5px;
}}
.doc h3 {{ font-size: 17px; color: var(--ink-soft); margin: 26px 0 10px; }}
.doc h4 {{ font-size: 15px; color: var(--muted); margin: 20px 0 8px; }}
.doc p {{ margin: 10px 0; }}
.doc a {{ color: var(--gold); text-decoration: none; border-bottom: 1px solid var(--gold-light); }}
.doc a:hover {{ border-bottom-color: var(--gold); }}
.doc ul, .doc ol {{ padding-left: 26px; margin: 10px 0; }}
.doc li {{ margin: 5px 0; }}
.doc code {{
  background: var(--code-bg); color: #8a5a2a; padding: 2px 6px;
  border-radius: 5px; font-size: 13.5px; font-family: Consolas, Monaco, monospace;
}}
.doc pre {{
  background: #2b2118; color: #e8dfd0; padding: 16px 20px; border-radius: 10px;
  overflow-x: auto; font-size: 13.5px; line-height: 1.6;
}}
.doc pre code {{ background: none; color: inherit; padding: 0; }}
.doc table {{
  border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px;
}}
.doc th, .doc td {{ border: 1px solid var(--border); padding: 8px 12px; text-align: left; }}
.doc th {{ background: var(--gold-pale); color: var(--ink-soft); font-weight: 600; }}
.doc tr:nth-child(even) td {{ background: #fdfbf7; }}
.doc blockquote {{
  border-left: 4px solid var(--gold-light); background: #fdfbf7;
  margin: 14px 0; padding: 10px 18px; color: var(--ink-soft);
}}
.doc blockquote p {{ margin: 4px 0; }}
.doc hr {{ border: none; border-top: 1px dashed var(--border); margin: 30px 0; }}
.doc img {{ max-width: 100%; border-radius: 8px; }}
.toc {{ font-size: 14px; }}
@media (max-width: 700px) {{ .doc {{ padding: 26px 20px; }} .wrap {{ padding: 20px 12px; }} }}
</style>
</head>
<body>
<div class="wrap"><article class="doc">
{body}
</article></div>
</body>
</html>
"""


def render(md_path: Path, out_path: Path | None = None) -> Path:
    text = md_path.read_text(encoding="utf-8")
    md = MarkdownIt("commonmark", {"html": True, "linkify": True}).enable("table").enable("strikethrough")
    body = md.render(text)
    title = md_path.stem
    out_path = out_path or md_path.with_suffix(".html")
    out_path.write_text(HTML_TEMPLATE.format(title=title, body=body), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=ROOT / "docs" / "使用手册.md")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = render(args.file, args.output)
    print(f"已渲染 → {out}")


if __name__ == "__main__":
    main()
