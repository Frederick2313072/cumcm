# 环境安装说明

## 方法一：使用requirements.txt（推荐）

```bash
# 1. 创建虚拟环境
python3 -m venv ai_music_env

# 2. 激活虚拟环境
source ai_music_env/bin/activate  # Linux/macOS
# 或
ai_music_env\Scripts\activate     # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python3 -c "import pandas, librosa, sklearn; print('所有依赖安装成功!')"
```

## 方法二：逐步安装

```bash
pip install pandas==2.3.1 openpyxl==3.1.5
pip install librosa==0.11.0 soundfile==0.13.1
pip install scikit-learn==1.7.1 matplotlib==3.10.5
pip install numpy==2.2.6 scipy==1.16.1 numba==0.61.2
```

## 运行方法

```bash
# 激活环境
source ai_music_env/bin/activate

# 处理附件音频文件
python3 process_attachments.py

# 或使用一键检测脚本
python3 val.py
```

## 注意事项

- Python版本要求：3.8+
- 推荐使用Python 3.10或3.11以获得最佳兼容性
- macOS用户可能需要额外安装音频处理系统依赖