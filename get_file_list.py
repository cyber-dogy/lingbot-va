#!/usr/bin/env python3
"""获取 HuggingFace 数据集所有文件的下载链接"""
import os
import sys

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import HfApi, hf_hub_url

def get_dataset_files(repo_id, repo_type="dataset", revision="main"):
    """获取数据集所有文件列表"""
    api = HfApi()
    
    print(f"正在获取文件列表: {repo_id}")
    print(f"镜像: {os.environ.get('HF_ENDPOINT', '默认')}")
    
    try:
        # 列出所有文件
        files = list(api.list_repo_files(repo_id, repo_type=repo_type, revision=revision))
        print(f"找到 {len(files)} 个文件")
        
        # 生成下载 URL
        urls = []
        for file in files:
            if file.endswith('/'):  # 跳过目录
                continue
            url = hf_hub_url(repo_id, file, repo_type=repo_type, revision=revision)
            # 替换为镜像地址
            url = url.replace('https://huggingface.co', 'https://hf-mirror.com')
            urls.append(url)
        
        return urls
    except Exception as e:
        print(f"错误: {e}")
        return []

if __name__ == "__main__":
    repo_id = "robbyant/robotwin-clean-and-aug-lerobot"
    
    urls = get_dataset_files(repo_id)
    
    if urls:
        output_file = "/home/gjw/MyProjects/lingbot-va/urls.txt"
        with open(output_file, 'w') as f:
            for url in urls:
                f.write(url + '\n')
        print(f"\n✓ URL 列表已保存到: {output_file}")
        print(f"共 {len(urls)} 个文件")
        
        # 显示前 5 个文件
        print("\n前 5 个文件:")
        for url in urls[:5]:
            print(f"  - {url.split('/')[-1]}")
    else:
        print("获取文件列表失败")
        sys.exit(1)
