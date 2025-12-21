# FLUX.2-dev API 使用指南

## API 端点信息

你的代码已经实现了 FastAPI 端点，支持**文生图**和**图生图**两种模式，可以通过 HTTP POST 请求调用。

### 端点 URL

部署后，Modal 会生成一个类似这样的 URL：
```
https://longlikun--example-flux2-web-dev.modal.run
```

> **注意**: 实际 URL 会在你运行 `modal deploy flux2.py` 后显示

## API 请求参数

### 请求体 (JSON)

```json
{
  "api_key": "longlikun",
  "prompt": "图片描述文本",
  "input_images": ["base64_encoded_image..."],
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 50,
  "guidance_scale": 4.0
}
```

### 参数说明

| 参数 | 类型 | 是否必需 | 默认值 | 说明 |
|------|------|----------|--------|------|
| `api_key` | string | **是** | 无 | API 密钥（固定值：`longlikun`）|
| `prompt` | string | 否 | "A cinematic photo of a baby penguin" | 图片生成提示词 |
| `input_images` | array[string] | 否 | `null` | **base64 编码的输入图片列表**，用于图生图模式（最多10张）|
| `width` | integer | 否 | 1024 | 图片宽度（像素）|
| `height` | integer | 否 | 1024 | 图片高度（像素）|
| `num_inference_steps` | integer | 否 | 50 | 推理步数（28-50，步数越多质量越好但速度越慢）|
| `guidance_scale` | float | 否 | 4.0 | 引导强度（FLUX.2 推荐值为 4.0）|

---

## 使用模式

### 模式 1: 文生图（Text-to-Image）

当 `input_images` 为 `null` 或不提供时，API 将从文本提示生成全新图片。

### 模式 2: 图生图（Image-to-Image）

当提供 `input_images` 时，API 将基于参考图片和文本提示进行图像编辑/转换。

支持的应用场景：
- 🎨 风格转换（如转为动漫风格、水彩画等）
- ✨ 图像增强和修改
- 🔄 保持角色/产品一致性
- 🖼️ 多图参考生成（最多10张）

---

## cURL 请求示例

### 文生图示例

```bash
curl -X POST https://longlikun--example-flux2-web-dev.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "longlikun",
    "prompt": "A futuristic cityscape at sunset"
  }' \
  --output output.jpg
```

### 图生图示例（单图）

```bash
# 步骤 1: 将图片编码为 base64
IMAGE_BASE64=$(base64 -i input.jpg)

# 步骤 2: 发送请求
curl -X POST https://longlikun--example-flux2-web-dev.modal.run \
  -H "Content-Type: application/json" \
  -d "{
    \"api_key\": \"longlikun\",
    \"prompt\": \"Transform into Studio Ghibli animation style\",
    \"input_images\": [\"$IMAGE_BASE64\"]
  }" \
  --output ghibli_style.jpg
```

---

## Python 请求示例

### 文生图

```python
import requests
from pathlib import Path

url = "https://longlikun--example-flux2-web-dev.modal.run"

payload = {
    "api_key": "longlikun",
    "prompt": "A serene Japanese garden with cherry blossoms",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 4.0
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    Path("output.jpg").write_bytes(response.content)
    print("✅ 图片已生成")
else:
    print(f"❌ 错误: {response.status_code}")
```

### 图生图

```python
import requests
import base64
from pathlib import Path

url = "https://longlikun--example-flux2-web-dev.modal.run"

# 读取并编码输入图片
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# 准备请求
payload = {
    "api_key": "longlikun",
    "prompt": "Transform this photo into a watercolor painting",
    "input_images": [
        encode_image("input.jpg")  # 单图
    ],
    "num_inference_steps": 50,
    "guidance_scale": 4.0
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    Path("watercolor.jpg").write_bytes(response.content)
    print("✅ 图片已生成")
else:
    print(f"❌ 错误: {response.status_code}")
    print(response.text)
```

### 图生图（多图参考）

```python
import requests
import base64
from pathlib import Path

url = "https://longlikun--example-flux2-web-dev.modal.run"

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

payload = {
    "api_key": "longlikun",
    "prompt": "Generate a new scene combining these character styles",
    "input_images": [
        encode_image("character1.jpg"),
        encode_image("character2.jpg"),
        encode_image("background.jpg")
    ],  # 多图参考，保持风格一致性
    "num_inference_steps": 50,
    "guidance_scale": 4.0
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    Path("combined.jpg").write_bytes(response.content)
    print("✅ 多图合成完成")
else:
    print(f"❌ 错误: {response.status_code}")
```

---

## JavaScript/Node.js 示例

### 图生图

```javascript
const fs = require('fs');
const fetch = require('node-fetch');

async function imageToImage() {
  const url = 'https://longlikun--example-flux2-web-dev.modal.run';
  
  // 读取并编码图片
  const imageBuffer = fs.readFileSync('input.jpg');
  const base64Image = imageBuffer.toString('base64');
  
  const payload = {
    api_key: 'longlikun',
    prompt: 'Convert to anime art style',
    input_images: [base64Image],
    num_inference_steps: 50,
    guidance_scale: 4.0
  };

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload)
    });

    if (response.ok) {
      const buffer = await response.buffer();
      fs.writeFileSync('anime_style.jpg', buffer);
      console.log('✅ 图片已转换');
    } else {
      console.error('❌ 错误:', response.status);
      const error = await response.json();
      console.error('详情:', error);
    }
  } catch (error) {
    console.error('❌ 错误:', error);
  }
}

imageToImage();
```

---

## 部署 API

### 开发模式（临时 URL）

```bash
modal serve flux2.py
```

这会生成一个临时的开发 URL，用于测试。

### 生产部署（永久 URL）

```bash
modal deploy flux2.py
```

这会生成一个稳定的生产 URL，可以长期使用。

---

## 响应格式

- **成功**: 返回 JPEG 图片文件（二进制数据）
  - Content-Type: `image/jpeg`
  - HTTP 状态码: 200

- **失败**: 返回错误信息
  - HTTP 状态码: 401 (无效密钥) 或 400/500 (其他错误)

---

## 性能提示

1. **推理步数权衡**:
   - 28 步: 更快，质量略低
   - 50 步: 官方推荐，质量最佳
   
2. **图片尺寸**:
   - 标准: 1024x1024
   - 宽屏: 1920x1080
   - 竖屏: 1080x1920

3. **输入图片优化**:
   - 建议在编码前将图片调整到目标分辨率
   - 压缩图片以减少 base64 payload 大小
   - 单张图片建议不超过 2MB（编码前）

4. **成本优化**:
   - H200 GPU 按秒计费
   - 图生图通常比文生图稍快（因为有参考）
   - 减少推理步数可降低单次请求成本

---

## 注意事项

> [!IMPORTANT]
> - 确保已设置 `huggingface-secret` Modal Secret
> - 首次调用会加载模型，可能需要较长时间（1-2分钟）
> - 后续调用会重用已加载的模型，速度更快
> - base64 编码会增加约 33% 的数据量

> [!TIP]
> - **文生图**: 使用明确、详细的提示词获得更好结果
> - **图生图**: 提示词应着重描述想要的**变化**，而非重复描述原图内容
> - **多图参考**: 最多支持 10 张参考图，适合保持风格/角色一致性

> [!WARNING]
> base64 编码的图片会显著增加 JSON payload 大小。一张 1MB 的图片编码后约为 1.33MB。建议在上传前压缩图片。
