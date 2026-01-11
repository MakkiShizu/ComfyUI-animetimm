# ComfyUI-animetimm

[English](README.md)

ComfyUI 自定义节点，使用 [animetimm](https://huggingface.co/animetimm) 的模型对动漫图像进行标签分类。

## 功能

- 使用多种 TIMM 模型进行动漫图像标签预测，来自[animetimm](https://huggingface.co/animetimm)
- 支持通用标签、角色标签、艺术家标签和分级标签的分类
- 支持批次图像处理
  ![image](Anime%20TIMM%20Classifier%20Example.png)

## 安装

### 推荐

- 通过 [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) 安装。

### 手动

- 在终端（cmd）中导航到 `ComfyUI/custom_nodes`。
- 使用以下命令在 `custom_nodes` 目录下克隆存储库：
  ```
  git clone https://github.com/MakkiShizu/ComfyUI-animetimm
  cd ComfyUI-animetimm
  ```
- 在你的 Python 环境中安装依赖项。
  - 对于 Windows 便携版，在 `ComfyUI\custom_nodes\ComfyUI-animetimm`内运行以下命令：
    ```
    ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
    ```
  - 如果使用 venv 或 conda，请先激活您的 Python 环境，然后运行：
    ```
    pip install -r requirements.txt
    ```

## 使用方法

### Anime TIMM Classifier

节点提供以下参数：

- `image`: 输入图像
- `threshold`: 置信度阈值（默认 0.35），置信度高于此值的标签才会被输出
- `model_repo`: 选择模型仓库（默认 animetimm/caformer_s36.dbv4-full）
- `include_general`: 是否包含通用标签（默认 True）
- `include_character`: 是否包含角色标签（默认 True）
- `include_artist`: 是否包含艺术家标签（默认 False）
- `include_rating`: 是否包含分级标签（默认 True）
- `replace_underscore`: 是否替换标签中的下划线为空格（默认 True）
- `use_custom_threshold`: 是否只使用自定义阈值（完全忽视 selected_tags.csv 中提供的 best_threshold）（默认为 False）

### 输出

- `tags`: 逗号分隔的标签字符串
- `confidence_scores`: 对应的置信度分数列表
- `raw_output`: 带有分类和置信度的原始输出
- `general_tags`: 通用标签
- `character_tags`: 角色标签
- `artist_tags`: 艺术家标签
- `rating_tags`: 分级标签

## 注意事项

- 首次运行时需要下载模型文件，可能需要一些时间
- 模型在第一次使用后会被缓存到 `ComfyUI/models/animetimm` 目录
- 推荐使用 GPU 进行推理以获得更好的性能
- 不同模型在精度和速度上有权衡，可根据需要选择
- 某些模型（如 `eva02_large_patch14_448.dbv4-full`）需要较大的显存

## 性能

ranklist:[dbv4-full-ranklist](https://huggingface.co/spaces/animetimm/dbv4-full-ranklist)
![image](ranklist.png)

## 常见问题

### 模型下载失败

官方仓库对公众开放，但您必须登录并接受相关条件才能访问其文件和内容。如果从官方仓库下载失败，节点会自动尝试从备份仓库下载。
