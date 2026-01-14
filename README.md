# Audio Loader from URL - ComfyUI 自定义节点

从 URL 加载音频文件的 ComfyUI 自定义节点。

## 功能特性

- ✅ 支持从 HTTP/HTTPS URL 下载音频文件
- ✅ 支持多种音频格式: MP3, WAV, OGG, M4A, FLAC, AAC, WMA
- ✅ 自动下载并缓存到临时文件
- ✅ 支持采样率转换
- ✅ 支持单声道/立体声转换
- ✅ 支持音频归一化
- ✅ 多后端支持: librosa, soundfile, pydub

## 安装

1. 将此文件夹复制到 ComfyUI 的`custom_nodes`目录
2. 安装依赖:

```bash
pip install -r requirements.txt
```

## 节点说明

### Audio Loader from URL (从 URL 加载音频)

从指定的 URL 加载音频文件并转换为波形数据。

**输入参数:**

- `url` (STRING): 音频文件的 URL 地址
- `sample_rate` (INT): 目标采样率,范围 8000-48000,默认 22050
- `channels` (STRING): 声道设置,可选 "mono" 或 "stereo",默认 "mono"
- `normalize` (BOOLEAN): 是否归一化音频到[-1, 1],默认 True

**输出:**

- `audio` (AUDIO): 音频波形数据,shape 为(1, samples)或(2, samples)的 torch tensor

### Audio Info Extractor (音频信息提取)

提取音频的基本信息。

**输入参数:**

- `audio` (AUDIO): 音频波形数据

**输出:**

- `duration` (INT): 音频时长(秒)
- `sample_rate` (INT): 采样率(Hz)
- `channels` (INT): 声道数
- `info` (STRING): 格式化的信息字符串

## 使用示例

### 基本用法

1. 添加 "Audio Loader from URL" 节点
2. 输入音频 URL,例如: `https://example.com/audio.mp3`
3. 设置采样率和声道
4. 连接到其他音频处理节点

### 支持的 URL 格式

- HTTP: `http://example.com/audio.mp3`
- HTTPS: `https://example.com/audio.wav`
- CDN 链接: `https://cdn.example.com/audio.ogg`
- 直接触发下载并处理

## 技术细节

### 支持的后端

节点会按以下顺序尝试使用音频处理库:

1. **librosa** (推荐): 功能最全,支持重采样
2. **soundfile**: 轻量级,速度快
3. **pydub**: 支持格式最多,但需要额外依赖

### 音频格式

输出格式:

- 单声道: shape 为 `(1, samples)`
- 立体声: shape 为 `(2, samples)`
- 数据类型: `torch.float32`
- 数值范围: [-1.0, 1.0] (归一化后)

### 注意事项

1. URL 必须是可公开访问的音频文件
2. 下载大文件可能需要较长时间
3. 临时文件会在加载后自动删除
4. 建议使用 HTTPS 协议以获得更好的安全性

## 依赖库

- `requests`: HTTP 请求
- `librosa`: 音频处理(推荐)
- `soundfile`: 音频文件读写
- `numpy`: 数值计算
- `torch`: PyTorch(ComfyUI 环境自带)

## 故障排除

### "需要安装音频处理库"

确保已安装至少一个音频处理库:

```bash
pip install librosa
# 或
pip install soundfile
```

### "下载音频失败"

- 检查 URL 是否正确
- 确保网络连接正常
- 检查 URL 是否可以公开访问
- 确认 URL 指向的是音频文件而不是网页

### "加载音频失败"

- 检查文件格式是否支持
- 确认音频文件没有损坏
- 尝试安装不同的音频处理库

## 许可证

MIT License

## 作者

ComfyUI Custom Nodes
