import os
import re
import argparse
import requests

def process_markdown(input_path, output_dir, article_name):
    # 读取原始 Markdown 文本
    with open(input_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
    
    # 1. 提取所有语雀图片URL
    pattern = re.compile(r'!\[.*?\]\((https://cdn\.nlark\.com/yuque/[^)]+)\)')
    img_urls = pattern.findall(md_text)
    
    # 准备输出路径
    posts_dir = os.path.join(output_dir, 'posts')
    images_dir = os.path.join(output_dir, 'assets', 'images', article_name)
    os.makedirs(posts_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # 2. 下载每个图片到本地
    for url in img_urls:
        # 去除URL中的锚点或查询参数以获取文件名和下载URL
        download_url = url.split('#')[0]    # 移除#及后面部分
        download_url = download_url.split('?')[0]   # 若有?参数也移除
        filename = os.path.basename(download_url)
        # 下载图片
        try:
            resp = requests.get(download_url, timeout=10)
            if resp.status_code == 200:
                img_path = os.path.join(images_dir, filename)
                with open(img_path, 'wb') as img_file:
                    img_file.write(resp.content)
                print(f"[下载成功] {url} -> {img_path}")
            else:
                print(f"[下载失败] {url}，HTTP状态码: {resp.status_code}")
        except Exception as e:
            print(f"[下载异常] 无法下载 {url} ，原因: {e}")
    
    # 3. 替换 Markdown 文本中的图片链接为本地路径
    def replace_link(match):
        original_url = match.group(1)
        # 获取对应的文件名
        fname = os.path.basename(original_url.split('#')[0].split('?')[0])
        # 构造相对路径引用，根据假定结构，上两级到assets
        new_path = f'../../assets/images/{article_name}/{fname}'
        return f'![]({new_path})'
    
    new_md_text = pattern.sub(replace_link, md_text)
    
    # 4. 移除语雀特有的反斜杠转义（如 \* \_）
    new_md_text = re.sub(r'\\([$*_])', r'\1', new_md_text)
    new_md_text = (
        new_md_text
        .replace('_$', '$')
        .replace(' $', '$')
        .replace('$ ', '$')
        )
    
    # 5. 输出清理后的 Markdown 文件
    output_md_path = os.path.join(posts_dir, f"{article_name}.md")
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(new_md_text)
    print(f"[处理完成] 已生成清洗后的 Markdown 文件: {output_md_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清理语雀导出的 Markdown，下载图片并修正链接")
    parser.add_argument("input_markdown", help="语雀导出的原始 Markdown 文件路径")
    parser.add_argument("output_dir", help="输出根目录（例如 mkdocs 的 docs 文件夹）")
    parser.add_argument("article_name", help="文章名称，将用于输出文件名和图片子目录")
    args = parser.parse_args()
    process_markdown(args.input_markdown, args.output_dir, args.article_name)
