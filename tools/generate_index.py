import os, re, yaml
from collections import defaultdict
from datetime import datetime
import pathlib

DOCS_DIR = pathlib.Path("docs")
POSTS_DIR = DOCS_DIR / "posts"
TAGS_DIR  = DOCS_DIR / "tags"
POSTS_INDEX_MD_PATH = POSTS_DIR /  "index.md"
TAGS_INDEX_MD_PATH = TAGS_DIR / "index.md"

FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)

def generate_post_index():
    def make_title_from_filename(filename):
        name = filename[:-3]  # 去掉 .md
        return name

    if not os.path.exists(POSTS_DIR):
        print(f"❌ 找不到目录: {POSTS_DIR}，请先创建并添加至少一篇文章")
        exit(1)

    all_posts = sorted(
        f for f in os.listdir(POSTS_DIR)
        if f.endswith(".md") and f != "index.md"
    )

    lines = [
        "# 📚 所有博文\n",
        "欢迎浏览我撰写的所有文章。\n",
        "---\n",
        "## 🔖 文章列表\n",
    ]

    for filename in all_posts:
        title = make_title_from_filename(filename)
        line = f"- [{title}]({filename})"
        lines.append(line)

    with open(POSTS_INDEX_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ 成功生成 {POSTS_INDEX_MD_PATH}，共收录 {len(all_posts)} 篇文章")

def generate_tag_index():
    def parse_front_matter(md_text):
        m = FRONT_MATTER_RE.match(md_text)
        if not m:
            return {}, md_text
        data = yaml.safe_load(m.group(1)) or {}
        body = md_text[m.end():]
        return data, body

    def slug(s):
        s = re.sub(r"\s+", "-", s.strip())
        s = re.sub(r"[^0-9A-Za-z\-\u4e00-\u9fff]", "", s)  # 保留中英文、数字、连字符
        return s.lower()

    def read_posts():
        posts = []
        for p in sorted(POSTS_DIR.rglob("*.md")):
            text = p.read_text(encoding="utf-8")
            meta, _ = parse_front_matter(text)
            tags = meta.get("tags") or []
            title = meta.get("title") or p.stem
            date  = meta.get("date")
            # 标准化日期
            try:
                dt = datetime.fromisoformat(str(date)) if date else None
            except Exception:
                dt = None
            posts.append({
                "path": p.relative_to(DOCS_DIR).as_posix(),
                "title": title,
                "tags": [str(t) for t in tags],
                "date": dt,
            })
        return posts

    def ensure_dir(d):
        d.mkdir(parents=True, exist_ok=True)

    def write_tag_pages(posts):
        ensure_dir(TAGS_DIR)
        by_tag = defaultdict(list)
        for post in posts:
            for t in post["tags"]:
                by_tag[t].append(post)

        # 标签索引页
        index = ["# 🏷️ 标签索引", "", "点击标签查看对应的文章。", ""]
        tags_sorted = sorted(by_tag.keys(), key=lambda x: slug(x))
        for t in tags_sorted:
            index.append(f"- [{t}](./{slug(t)}.md) ({len(by_tag[t])})")
        (TAGS_DIR / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")

        # 各标签页
        for t in tags_sorted:
            items = sorted(by_tag[t], key=lambda x: x["date"] or datetime.min, reverse=True)
            lines = [f"# 标签：{t}", ""]
            for it in items:
                date_str = it["date"].date().isoformat() if it["date"] else ""
                title = it["title"]
                path  = it["path"]
                lines.append(f"- {date_str} [{title}](../{path})")
            (TAGS_DIR / f"{slug(t)}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    posts = read_posts()
    write_tag_pages(posts)
    print(f"✅ 已生成标签页到 {TAGS_DIR}")

if __name__ == "__main__":
    generate_post_index()
    generate_tag_index()