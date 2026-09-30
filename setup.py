# -*- coding: utf-8 -*-
"""DeepSearch-RL 包安装脚本。

核心训练依赖（torch / verl / sglang）需按 README 单独安装；本包安装工程自身代码与
通用依赖。
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="deepsearch-rl",
    version="0.1.0",
    description="DeepSearch-RL：基于 Tool-Agentic RL 的多跳搜索智能体（Qwen3-8B / veRL / SGLang / GRPO）",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Simon11866",
    url="https://github.com/Simon11866/DeepSearch-RL",
    packages=find_packages(exclude=("tests", "tests.*", "examples")),
    python_requires=">=3.10",
    install_requires=[
        "modelscope==1.23.1",
        "datasets>=2.20.0",
        "huggingface_hub>=0.25.0",
        "pyarrow>=15.0.0",
        "transformers>=4.51.0",
        "accelerate>=0.34.0",
        "peft>=0.12.0",
        "hydra-core>=1.3.2",
        "numpy<2.3",
        "pandas",
        "swanlab==0.9.0",
        "openai>=1.40.0",
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "tenacity>=2.3.0",
        "aiohttp>=3.10.0",
        "requests>=2.32.0",
        "tiktoken",
        "trafilatura>=1.12.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=5.3.0",
        "liger-kernel>=0.8.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
