# n8n HTTP 节点配置指南 - FLUX.2 API

## 📌 概述

本指南详细说明如何在 n8n 中使用 HTTP Request 节点调用 FLUX.2-dev API，支持文生图和图生图两种模式。

---

## 方案 1: 文生图（Text-to-Image）

### HTTP Request 节点配置

#### 基本设置
- **Method**: `POST`
- **URL**: `https://rodneycornwell--example-flux2-web.modal.run`
- **Authentication**: `None`
- **Send Body**: `Yes`
- **Body Content Type**: `JSON`

#### Body/JSON 内容
```json
{
  "api_key": "longlikun",
  "prompt": "{{ $json.prompt }}",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 50,
  "guidance_scale": 4.0
}
```

#### 响应设置
- **Response Format**: `File`
- **Binary Property**: `data`
- **File Name**: `{{ $json.filename || 'flux2_output.jpg' }}`

### 工作流示例

```
[Manual Trigger] 
    ↓ 
    {
      "prompt": "A futuristic cityscape at sunset"
    }
    ↓
[HTTP Request] → 配置如上
    ↓
[Write Binary File] → 保存图片到磁盘
```

---

## 方案 2: 图生图（Image-to-Image）

### 步骤 1: 读取图片并转换为 Base64

#### Read Binary File 节点
- **File Path**: `{{ $json.image_path }}`
- **Property Name**: `image_data`

#### Code 节点（转换为 Base64）
```javascript
// 将二进制数据转换为 base64 字符串
const items = [];

for (const item of $input.all()) {
  const binaryData = item.binary.image_data;
  
  // 转换为 base64
  const base64String = Buffer.from(binaryData.data, 'base64').toString('base64');
  
  items.push({
    json: {
      ...item.json,
      image_base64: base64String
    }
  });
}

return items;
```

### 步骤 2: 调用 API

#### HTTP Request 节点配置

```json
{
  "api_key": "longlikun",
  "prompt": "{{ $json.edit_prompt }}",
  "input_images": ["{{ $json.image_base64 }}"],
  "num_inference_steps": 50,
  "guidance_scale": 4.0
}
```

### 完整工作流示例

```
[Manual Trigger]
    ↓
    {
      "image_path": "/path/to/input.jpg",
      "edit_prompt": "Transform into anime style"
    }
    ↓
[Read Binary File] 
    ↓
[Code: Convert to Base64]
    ↓
[HTTP Request - FLUX.2 API]
    ↓
[Write Binary File] → 保存生成的图片
```

---

## 方案 3: 从 URL 下载图片并转换

如果图片来源是 URL（例如从另一个 API 获取）：

### HTTP Request 节点 1: 下载图片
- **Method**: `GET`
- **URL**: `{{ $json.image_url }}`
- **Response Format**: `File`

### Code 节点: URL 图片转 Base64
```javascript
const items = [];

for (const item of $input.all()) {
  // 获取二进制数据
  const binaryKey = Object.keys(item.binary)[0];
  const binaryData = item.binary[binaryKey];
  
  // 转换为 base64
  const base64String = binaryData.data;
  
  items.push({
    json: {
      ...item.json,
      image_base64: base64String
    }
  });
}

return items;
```

### HTTP Request 节点 2: 调用 FLUX.2 API
```json
{
  "api_key": "longlikun",
  "prompt": "{{ $json.prompt }}",
  "input_images": ["{{ $json.image_base64 }}"],
  "num_inference_steps": 50
}
```

---

## 🎯 实用工作流模板

### 模板 1: 批量文生图

```
[Spreadsheet File] 读取 CSV（包含多个 prompt）
    ↓
[Split In Batches] 每次处理 1 条
    ↓
[HTTP Request] 调用 FLUX.2 API
    ↓
[Write Binary File] 保存到文件夹
```

### 模板 2: 图片风格转换流水线

```
[Webhook] 接收图片 URL + 风格描述
    ↓
[HTTP Request] 下载原始图片
    ↓
[Code] 转换为 base64
    ↓
[HTTP Request] 调用 FLUX.2 图生图 API
    ↓
[HTTP Request] 上传到云存储
    ↓
[Webhook Response] 返回生成图片 URL
```

### 模板 3: 多图参考生成

```javascript
// Code 节点：处理多张输入图片
const items = [];

for (const item of $input.all()) {
  const base64Images = [];
  
  // 假设有 3 张图片
  for (let i = 1; i <= 3; i++) {
    const key = `image${i}`;
    if (item.binary[key]) {
      base64Images.push(item.binary[key].data);
    }
  }
  
  items.push({
    json: {
      prompt: item.json.prompt,
      input_images: base64Images
    }
  });
}

return items;
```

---

## ⚙️ 高级配置

### 错误处理

在 HTTP Request 节点中添加错误处理：

**Settings** → **Options**:
- ✅ `Continue On Fail`: Enabled
- ✅ `Retry On Fail`: 3 times
- ✅ `Wait Between Tries`: 5000 ms

### 超时设置

由于图片生成可能需要较长时间：

**Settings** → **Options**:
- `Timeout`: `300000` (5分钟)

### 响应验证

使用 **IF 节点** 检查响应：

```javascript
// 检查是否成功返回图片
{{ $json.statusCode === 200 }}
```

---

## 📝 完整示例：n8n 工作流 JSON

### 文生图工作流

```json
{
  "nodes": [
    {
      "parameters": {},
      "name": "Manual Trigger",
      "type": "n8n-nodes-base.manualTrigger",
      "position": [240, 300]
    },
    {
      "parameters": {
        "url": "https://rodneycornwell--example-flux2-web.modal.run",
        "method": "POST",
        "sendBody": true,
        "contentType": "json",
        "bodyParameters": {
          "parameters": [
            {
              "name": "api_key",
              "value": "longlikun"
            },
            {
              "name": "prompt",
              "value": "A futuristic cityscape at sunset"
            },
            {
              "name": "num_inference_steps",
              "value": 50
            }
          ]
        },
        "options": {
          "response": {
            "response": {
              "responseFormat": "file"
            }
          },
          "timeout": 300000
        }
      },
      "name": "FLUX2 API",
      "type": "n8n-nodes-base.httpRequest",
      "position": [440, 300]
    },
    {
      "parameters": {
        "fileName": "flux2_output.jpg",
        "dataPropertyName": "data"
      },
      "name": "Save Image",
      "type": "n8n-nodes-base.writeBinaryFile",
      "position": [640, 300]
    }
  ],
  "connections": {
    "Manual Trigger": {
      "main": [[{"node": "FLUX2 API", "type": "main", "index": 0}]]
    },
    "FLUX2 API": {
      "main": [[{"node": "Save Image", "type": "main", "index": 0}]]
    }
  }
}
```

---

## 💡 最佳实践

### 1. Base64 性能优化
- 压缩图片到目标分辨率（1024x1024）再编码
- 避免上传超大图片（建议 < 2MB）

### 2. 批量处理
- 使用 `Split In Batches` 节点控制并发
- 每批次建议 1-3 张图片

### 3. 成本控制
- 缓存常用的生成结果
- 使用较低的 `num_inference_steps`（28-35）进行预览

### 4. 错误处理
- 总是启用 `Continue On Fail`
- 记录失败的 prompt 和参数

---

## 🔍 调试技巧

### 查看 Base64 编码
在 Code 节点后添加：
```javascript
return [{
  json: {
    base64_length: $json.image_base64.length,
    first_100_chars: $json.image_base64.substring(0, 100)
  }
}];
```

### 测试 API 连接
简单的 ping 测试：
```json
{
  "api_key": "longlikun",
  "prompt": "test"
}
```

---

## ❓ 常见问题

**Q: 图片太大导致超时？**  
A: 在上传前使用 Image Resize 节点或压缩图片。

**Q: Base64 编码失败？**  
A: 确保二进制数据属性名称正确，使用 `$binary.data` 访问。

**Q: n8n 中如何处理多张图片？**  
A: 使用 Code 节点遍历多个二进制属性并构建数组。

**Q: 如何在 n8n 中预览生成的图片？**  
A: 使用 Write Binary File 节点保存，或发送到 Webhook 节点返回。
