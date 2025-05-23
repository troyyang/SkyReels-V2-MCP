[English](README.md)

---

# 🎬 SkyReels-V2 MCP

## 项目简介

**SkyReels-V2 MCP** 是 SkyReels-V2 的增强版本，提供了一套完整的视频生成解决方案，支持客户端与服务器端的交互操作。你可以使用它实现资源上传、视频生成、结果下载等功能，适用于自动化短视频内容生成等场景。

* 📁 **客户端**：支持文件上传、视频生成、下载结果。
* 🖥️ **服务器端**：支持资源读取、资源列表、调用生成工具等功能。

---

## 🌟 示例演示

<table>
  <tr>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=-wd-8d6ShuY">
        <img src="https://github.com/troyyang/assets/raw/main/SkyReels-V2-MCP/thumb-young.png" width="100%"/>
        <p>年轻人动画视频</p>
      </a>
    </td>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=uDk81mYhsrM">
        <img src="https://github.com/troyyang/assets/raw/main/SkyReels-V2-MCP/thumb-swans.png" width="100%"/>
        <p>天鹅视频示例</p>
      </a>
    </td>
  </tr>
</table>

---

## 📦 安装指南

请确保系统已安装以下依赖：

* `git`
* [`uv`](https://github.com/astral-sh/uv)（用于Python项目管理）

### 一键安装

在终端执行以下命令：

```bash
./install.sh
```

### 安装脚本功能

`install.sh` 脚本会自动完成以下操作：

1. 检查 `git` 与 `uv` 是否已安装；
2. 克隆 SkyReels-V2 仓库；
3. 拷贝必要项目文件；
4. 执行 `uv sync` 同步依赖；
5. 使用 `uv pip` 安装额外依赖项（如 `torch` 和 `flash-attn`）。

---

## 🚀 启动服务端

服务端负责处理客户端的所有请求。

### 启动命令：

```bash
uv run server.py
```

### 功能一览：

* ✅ 支持读取本地文件资源（`file://` URI）
* ✅ 查询上传目录下的文件列表
* ✅ 视频生成、文件上传等工具调用

---

## 🖥️ 启动客户端

客户端提供便捷的交互操作，包括上传素材、生成视频、下载文件等。

### 启动命令：

```bash
uv run client.py
```

### 功能一览：

* ⬆️ 文件上传至服务器
* 🎥 基于文本提示 + 图像生成视频
* ⬇️ 下载生成的视频文件到本地目录

---

## 🔗 项目地址

* 🌐 原始项目：[SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2)
* 📦 本项目仓库：[SkyReels-V2 MCP](https://github.com/troyyang/SkyReels-V2-MCP)

---

## 📬 联系作者

如有问题或合作意向，欢迎通过微信联系：

<img src="https://github.com/troyyang/assets/raw/main/SkyReels-V2-MCP/wechat.png" width="180"/>

---

如果你觉得本项目有帮助，欢迎 ⭐Star 支持！

---