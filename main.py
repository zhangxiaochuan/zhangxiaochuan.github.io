# main.py
import re
from urllib.parse import quote

def define_env(env):
    def slug(s: str) -> str:
        s = s.strip()
        s = re.sub(r"\s+", "-", s)                             # 空白→-
        s = re.sub(r"[^0-9A-Za-z\-\u4e00-\u9fff]", "", s)      # 仅保留中英文/数字/-
        return s.lower()

    @env.macro
    def render_tags(items=None, title="标签"):
        # 取文章 front matter 中的 tags
        items = items or env.page.meta.get("tags", [])
        if not items:
            return ""
        chips = []
        for t in items:
            s = slug(str(t))
            href = f"/tags/{s}/"             # 根路径跳转，与你的聚合页保持一致
            chips.append(
                f'<a class="mmf-tag" href="{href}" title="{t}">{t}</a>'
            )
        return f'''
<div class="mmf-tags">
  <span class="mmf-tags-title">{title}：</span>
  {"".join(chips)}
</div>
'''.strip()
