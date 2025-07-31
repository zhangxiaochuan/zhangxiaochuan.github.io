import os

POSTS_DIR = "docs/posts"
INDEX_MD_PATH = os.path.join(POSTS_DIR, "index.md")

def make_title_from_filename(filename):
    name = filename[:-3]  # 去掉 .md
    name = name.replace("-", " ")
    return name.title()

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

with open(INDEX_MD_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"✅ 成功生成 {INDEX_MD_PATH}，共收录 {len(all_posts)} 篇文章")
