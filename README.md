<header>

<!--
  <<< Author notes: Course header >>>
  Include a 1280×640 image, course title in sentence case, and a concise description in emphasis.
  In your repository settings: enable template repository, add your 1280×640 social image, auto delete head branches.
  Add your open source license, GitHub uses MIT license.
-->

# Multimodal fine-tuning (Step 1)

_统一图片、文本、结构化数据的预处理与对齐流程，作为后续 LoRA + RAG 方案的第一步。_

## 目录

1. [多模态预处理概览](#多模态预处理概览)
2. [配置文件](#配置文件)
3. [数据清单 manifest](#数据清单-manifest)
4. [运行流水线](#运行流水线)
5. [输出结构](#输出结构)
6. [下一步建议](#下一步建议)

## 多模态预处理概览

`preprocessing/` 目录提供可复用的 Python 模块：

- `config.py`：定义图片、文本、结构化字段的配置结构，并支持从 JSON 配置初始化。
- `pipeline.py`：实现图像归一化（固定尺寸、均值方差）、文本 Tokenization（Qwen Tokenizer）、结构化特征标准化/one-hot，并生成统一 prompt。
- `config.example.json`：给出示例配置，可直接复制修改。
- `preprocess_dataset.py`：命令行入口，读取 JSONL manifest，输出 `.npz` tensor + metadata。

整体处理流程：

1. 读取 manifest 中的 `image_path`、`text`、`structured` 字段。
2. 对图片执行 resize、RGB 归一化，输出 `pixel_values (3×H×W)`。
3. 对文本执行 tokenizer，输出 `input_ids` 与 `attention_mask`。
4. 结构化字段支持连续值 z-score 以及离散值 one-hot，同时输出一段可直接拼接到 RAG Prompt 的说明文本。
5. 将多模态内容包装成统一 prompt：

   ```text
   <image>
   <structured>
   字段: 值
   </structured>
   <instruction>
   原始文本问题/指令
   </instruction>
   ```

该 prompt 可以直接送入 Qwen-7B 的指令模板，也方便在 RAG 阶段与检索上下文拼接。

## 配置文件

复制 `preprocessing/config.example.json`，按需修改：

```json
{
  "image": {
    "size": [224, 224],
    "mean": [0.48145466, 0.4578275, 0.40821073],
    "std": [0.26862954, 0.26130258, 0.27577711]
  },
  "text": {
    "tokenizer_name": "Qwen/Qwen-7B",
    "max_length": 512,
    "padding": false
  },
  "structured": {
    "fields": [
      {"name": "age", "kind": "continuous", "mean": 40.0, "std": 12.0},
      {"name": "gender", "kind": "categorical", "vocabulary": ["male", "female"]}
    ]
  }
}
```

- `continuous` 字段需要均值/方差，流水线会自动进行 z-score。
- `categorical` 字段会生成 one-hot，并保留 `default` 兜底值。

## 数据清单 manifest

使用 JSON Lines 文件描述多模态样本（每行一个 JSON 对象）：

```json
{"image_path": "dataset/img_0001.jpg", "text": "描述或问题", "structured": {"age": 33, "gender": "male"}}
```

建议在准备数据时就将图片路径、文本、结构化字段补齐，方便后续批量处理。

## 运行流水线

```bash
python preprocess_dataset.py \
  data/manifest.jsonl \
  preprocessing/config.example.json \
  processed/
```

脚本会：

1. 加载配置并实例化 `MultimodalPreprocessor`。
2. 对 manifest 中的每条数据执行对齐逻辑。
3. 将张量写入 `processed/sample_000000.npz` 等文件，同时在 `metadata.jsonl` 中保存 prompt、原始文本等信息。

## 输出结构

- `.npz`：包含 `pixel_values`、`input_ids`、`attention_mask`、`structured_vector`。
- `metadata.jsonl`：记录 `prompt`、`structured_prompt`、原始 `image_path` 与 `text`，便于手动抽样检查或构建下游 RAG 索引。

这些文件可以直接作为 LoRA/QLoRA 训练的数据来源，在 `DataLoader` 中读取 `.npz` 后即可拼接 batch。

## 下一步建议

1. 基于 `metadata.jsonl` 抽取代表性的 prompt 进行人工质检，确保多模态信息被完整地拼入模板。
2. 将 `structured_prompt` 文本片段同步写入后续的向量检索库，以便 RAG 在推理阶段检索同构字段。
3. 在完成数据对齐后，可以着手准备 LoRA 训练脚本，直接消费这些 `.npz` 文件。

# Introduction to GitHub

_Get started using GitHub in less than an hour._

</header>

<!--
  <<< Author notes: Step 1 >>>
  Choose 3-5 steps for your course.
  The first step is always the hardest, so pick something easy!
  Link to docs.github.com for further explanations.
  Encourage users to open new tabs for steps!
-->

## Step 1: Create a branch

_Welcome to "Introduction to GitHub"! :wave:_

**What is GitHub?**: GitHub is a collaboration platform that uses _[Git](https://docs.github.com/get-started/quickstart/github-glossary#git)_ for versioning. GitHub is a popular place to share and contribute to [open-source](https://docs.github.com/get-started/quickstart/github-glossary#open-source) software.
<br>:tv: [Video: What is GitHub?](https://www.youtube.com/watch?v=pBy1zgt0XPc)

**What is a repository?**: A _[repository](https://docs.github.com/get-started/quickstart/github-glossary#repository)_ is a project containing files and folders. A repository tracks versions of files and folders. For more information, see "[About repositories](https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories)" from GitHub Docs.

**What is a branch?**: A _[branch](https://docs.github.com/en/get-started/quickstart/github-glossary#branch)_ is a parallel version of your repository. By default, your repository has one branch named `main` and it is considered to be the definitive branch. Creating additional branches allows you to copy the `main` branch of your repository and safely make any changes without disrupting the main project. Many people use branches to work on specific features without affecting any other parts of the project.

Branches allow you to separate your work from the `main` branch. In other words, everyone's work is safe while you contribute. For more information, see "[About branches](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches)".

**What is a profile README?**: A _[profile README](https://docs.github.com/account-and-profile/setting-up-and-managing-your-github-profile/customizing-your-profile/managing-your-profile-readme)_ is essentially an "About me" section on your GitHub profile where you can share information about yourself with the community on GitHub.com. GitHub shows your profile README at the top of your profile page. For more information, see "[Managing your profile README](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-profile/customizing-your-profile/managing-your-profile-readme)".

![profile-readme-example](/images/profile-readme-example.png)

### :keyboard: Activity: Your first branch

1. Open a new browser tab and navigate to your newly made repository. Then, work on the steps in your second tab while you read the instructions in this tab.
2. Navigate to the **< > Code** tab in the header menu of your repository.

   ![code-tab](/images/code-tab.png)

3. Click on the **main** branch drop-down.

   ![main-branch-dropdown](/images/main-branch-dropdown.png)

4. In the field, name your branch `my-first-branch`. In this case, the name must be `my-first-branch` to trigger the course workflow.
5. Click **Create branch: my-first-branch** to create your branch.

   ![create-branch-button](/images/create-branch-button.png)

   The branch will automatically switch to the one you have just created.
   The **main** branch drop-down bar will reflect your new branch and display the new branch name.

6. Wait about 20 seconds then refresh this page (the one you're following instructions from). [GitHub Actions](https://docs.github.com/en/actions) will automatically update to the next step.

<footer>

<!--
  <<< Author notes: Footer >>>
  Add a link to get support, GitHub status page, code of conduct, license link.
-->

---

Get help: [Post in our discussion board](https://github.com/orgs/skills/discussions/categories/introduction-to-github) &bull; [Review the GitHub status page](https://www.githubstatus.com/)

&copy; 2024 GitHub &bull; [Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/code_of_conduct.md) &bull; [MIT License](https://gh.io/mit)

</footer>
