[中文](README_zh.md)

---

# 🎬 SkyReels-V2 MCP

## Project Overview

**SkyReels-V2 MCP** is an enhanced version of the original SkyReels-V2 project. It offers a complete video generation solution with both client and server components. The server handles resource access, tool execution, and video generation tasks, while the client supports file uploads, video generation requests, and downloading results.

---

## 🌟 Demo Previews
<table>
  <tr>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=-wd-8d6ShuY">
        <img src="https://github.com/troyyang/assets/raw/main/SkyReels-V2-MCP/thumb-young.png" width="100%"/>
        <p>Young Person Animation</p>
      </a>
    </td>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=uDk81mYhsrM">
        <img src="https://github.com/troyyang/assets/raw/main/SkyReels-V2-MCP/thumb-swans.png" width="100%"/>
        <p>Swans Video Sample</p>
      </a>
    </td>
  </tr>
</table>

---

## 📦 Installation Guide

Before getting started, ensure the following tools are installed on your system:

* `git`
* [`uv`](https://github.com/astral-sh/uv) (a modern Python project management tool)

### Quick Install

Run the following command in your terminal:

```bash
./install.sh
```

### What the Install Script Does

The `install.sh` script performs the following steps:

1. Checks for `git` and `uv`, and exits with an error message if they’re missing.
2. Clones the SkyReels-V2 repository.
3. Copies project files to the current working directory.
4. Runs `uv sync` to synchronize dependencies.
5. Installs required packages such as `torch` and `flash-attn` using `uv pip`.

---

## 🚀 Running the Server

The server is responsible for handling client requests, reading local resources, listing uploaded files, and calling video generation tools.

### Start Command

```bash
uv run server.py
```

### Server Capabilities

* 📂 Read local resources via `file://` URIs (HTTP/HTTPS not supported yet)
* 📋 List files in the upload directory
* 🧠 Handle tool execution such as file uploads and video generation

---

## 🖥️ Running the Client

Once the server is running, you can launch the client to interact with it—uploading files, generating videos, and downloading results.

### Start Command

```bash
uv run client.py
```

### Client Features

* ⬆️ Upload local files to the server
* 🎬 Generate videos from text prompts and optional image inputs
* ⬇️ Download generated video files to your local directory

---

## 🔗 Project Links

* 📌 Original SkyReels-V2: [SkyReels-V2 on GitHub](https://github.com/SkyworkAI/SkyReels-V2)
* 📦 MCP Version: [SkyReels-V2 MCP on GitHub](https://github.com/troyyang/SkyReels-V2-MCP)

---

## 📬 Contact

For questions, feedback, or collaboration, feel free to reach out via Whatsapp:

<img src="https://github.com/troyyang/assets/raw/main/SkyReels-V2-MCP/whatsapp.png" width="180"/>

---

If you find this project helpful, feel free to give it a ⭐ on GitHub!

---