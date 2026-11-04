# Skill: LaTeX 写作与 PDF 编译

当用户要求撰写 LaTeX 论文并编译为 PDF 时，使用本 skill。

---

## 环境安装（一次性）

在 Ubuntu/Debian 上执行（需要 sudo）：

```bash
# 配置代理（字节内网环境）
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

# 安装 TexLive 核心 + 常用宏包 + latexmk 自动编译
sudo apt-get update -qq
sudo apt-get install -y texlive-latex-base texlive-fonts-recommended texlive-latex-extra texlive-science latexmk
```

安装完成后 `pdflatex`、`bibtex`、`latexmk` 即可用。

---

## 编译命令

### 推荐：latexmk 一键编译（自动处理引用、多次编译）

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

### 手动编译（带 BibTeX 引用时）

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

### 清理中间文件

```bash
latexmk -C
```

---

## 最小示例

### 文件结构

```
my-paper/
├── main.tex       # 主文件
└── refs.bib       # 参考文献（可选）
```

### main.tex

```latex
\documentclass{article}
\usepackage{amsmath,graphicx,booktabs}
\usepackage[numbers]{natbib}

\title{Paper Title}
\author{Author Name}

\begin{document}
\maketitle

\begin{abstract}
Abstract text here.
\end{abstract}

\section{Introduction}
Diffusion models~\citep{ho2020ddpm} have shown remarkable results.

\section{Method}
The forward process:
\begin{equation}
q(x_t|x_0) = \mathcal{N}\bigl(\sqrt{\bar{\alpha}_t} x_0,\; (1-\bar{\alpha}_t)I\bigr)
\end{equation}

\section{Experiments}
\begin{table}[h]
\centering
\caption{Results comparison}
\begin{tabular}{lcc}
\toprule
Method & FID $\downarrow$ & IS $\uparrow$ \\
\midrule
Baseline & 3.60 & 240.1 \\
Ours & \textbf{2.45} & \textbf{268.3} \\
\bottomrule
\end{tabular}
\end{table}

\bibliographystyle{plainnat}
\bibliography{refs}

\end{document}
```

### refs.bib

```bibtex
@article{ho2020ddpm,
  author  = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  title   = {Denoising Diffusion Probabilistic Models},
  journal = {NeurIPS},
  year    = {2020}
}
```

### 编译

```bash
cd my-paper
latexmk -pdf -interaction=nonstopmode main.tex
# 产出: main.pdf
```

---

## Agent 工作流程

1. **写 .tex 文件** — 根据用户需求生成 `main.tex`（以及可选的 `refs.bib`）
2. **编译** — 执行 `latexmk -pdf -interaction=nonstopmode main.tex`
3. **检查结果** — 确认 PDF 文件存在且大小合理（`ls -la main.pdf`）
4. **处理错误** — 如编译失败，查看 `main.log` 中的错误信息，修复后重新编译

### 常见错误速查

| 错误 | 原因 | 修复 |
|------|------|------|
| `File 'xxx.sty' not found` | 缺少宏包 | `sudo apt-get install texlive-latex-extra` |
| `Missing $ inserted` | 数学符号在文本模式 | 用 `$...$` 包围 |
| `Undefined control sequence` | 命令未定义 | 检查 `\usepackage` 是否正确 |
| `Citation undefined` | 引用未解析 | 需要运行 bibtex（用 latexmk 自动处理） |

---

## NeurIPS 模板用法

```bash
# 下载 NeurIPS 2024 模板
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
wget https://media.neurips.cc/Conferences/NeurIPS2024/Styles/neurips_2024.sty

# 使用
# \documentclass{article}
# \usepackage{neurips_2024}
```

编译方式不变：`latexmk -pdf -interaction=nonstopmode main.tex`
