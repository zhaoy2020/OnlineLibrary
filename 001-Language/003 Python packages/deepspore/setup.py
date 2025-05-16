from setuptools import setup, find_packages


setup(
    name= "deepspore",                                  # 包名（PyPI唯一标识）
    version= "0.1.2",                                   # 版本号（遵循语义化版本）
    author= "Yu Zhao",
    author_email= "zhao_sy@126.com",
    description= "A simple tools for Deep Learning",
    long_description= open("README.md").read(),
    long_description_content_type= "text/markdown",
    packages= find_packages(),                          # 自动发现所有包
)