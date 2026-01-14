"""
ComfyUI自定义节点 - Audio Loader from URL
从URL加载音频文件并转换为波形数据
"""

import os
import tempfile
import requests
import torch
import numpy as np
from typing import Optional


class AudioLoaderFromURL:
    """
    从URL加载音频文件
    支持多种音频格式(mp3, wav, ogg, m4a等)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "sample_rate": ("INT", {
                    "default": 22050,
                    "min": 8000,
                    "max": 48000,
                    "step": 100
                }),
                "channels": (["mono", "stereo"], {
                    "default": "mono"
                }),
                "normalize": ("BOOLEAN", {
                    "default": True
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load_audio_from_url"
    CATEGORY = "audio/loader"
    
    def load_audio_from_url(self, url: str, sample_rate: int = 22050, 
                           channels: str = "mono", normalize: bool = True):
        """
        从URL加载音频文件
        
        Args:
            url: 音频文件的URL地址
            sample_rate: 目标采样率
            channels: 声道设置 ("mono" 或 "stereo")
            normalize: 是否归一化音频
            
        Returns:
            audio: 音频波形数据, shape为(1, samples)的torch tensor
        """
        
        if not url or not url.strip():
            raise ValueError("URL不能为空")
        
        # 下载音频文件
        print(f"正在从URL下载音频: {url}")
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # 获取文件扩展名
            content_type = response.headers.get('content-type', '')
            extension = self._get_audio_extension(url, content_type)
            
            # 创建临时文件保存音频
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name
            
            print(f"音频已下载到临时文件: {tmp_file_path}")
            
            # 读取音频文件
            audio_waveform = self._load_audio_file(
                tmp_file_path, 
                sample_rate, 
                channels, 
                normalize
            )
            
            # 删除临时文件
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            
            print(f"音频加载成功, shape: {audio_waveform.shape}")
            
            return (audio_waveform,)
            
        except requests.RequestException as e:
            raise ValueError(f"下载音频失败: {str(e)}")
        except Exception as e:
            raise ValueError(f"加载音频失败: {str(e)}")
    
    def _get_audio_extension(self, url: str, content_type: str) -> str:
        """
        根据URL或Content-Type推断音频文件扩展名
        
        Args:
            url: 音频URL
            content_type: HTTP响应的Content-Type
            
        Returns:
            extension: 文件扩展名(包含点号)
        """
        # 首先尝试从URL中提取
        url_lower = url.lower()
        for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac', '.wma']:
            if url_lower.endswith(ext):
                return ext
        
        # 然后尝试从Content-Type推断
        content_type_lower = content_type.lower()
        if 'mp3' in content_type_lower:
            return '.mp3'
        elif 'wav' in content_type_lower:
            return '.wav'
        elif 'ogg' in content_type_lower:
            return '.ogg'
        elif 'm4a' in content_type_lower:
            return '.m4a'
        elif 'flac' in content_type_lower:
            return '.flac'
        elif 'aac' in content_type_lower:
            return '.aac'
        elif 'wma' in content_type_lower:
            return '.wma'
        
        # 默认返回.wav
        return '.wav'
    
    def _load_audio_file(self, file_path: str, sample_rate: int, 
                        channels: str, normalize: bool) -> torch.Tensor:
        """
        加载音频文件并转换为波形数据
        
        Args:
            file_path: 音频文件路径
            sample_rate: 目标采样率
            channels: 声道设置
            normalize: 是否归一化
            
        Returns:
            waveform: 音频波形数据, shape为(1, samples)的torch tensor
        """
        try:
            # 尝试使用librosa
            import librosa
            import librosa.display
            
            # 加载音频
            audio, sr = librosa.load(
                file_path,
                sr=sample_rate,
                mono=(channels == "mono")
            )
            
            # 转换为torch tensor
            waveform = torch.from_numpy(audio).float()
            
            # 如果是立体声,添加通道维度
            if channels == "stereo":
                # librosa默认返回单声道,这里我们扩展为立体声
                # 注意: 如果源文件是单声道,这里会复制两份
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0).repeat(2, 1)
            else:
                # 单声道,添加batch维度
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
            
            # 归一化
            if normalize:
                max_val = torch.abs(waveform).max()
                if max_val > 0:
                    waveform = waveform / max_val
            
            return waveform
            
        except ImportError:
            print("librosa未安装,尝试使用soundfile...")
            try:
                import soundfile as sf
                
                # 使用soundfile读取
                audio, sr = sf.read(file_path)
                
                # 转换为torch tensor
                waveform = torch.from_numpy(audio).float()
                
                # 处理声道
                if waveform.ndim == 1:
                    # 单声道
                    if channels == "stereo":
                        waveform = waveform.unsqueeze(0).repeat(2, 1)
                    else:
                        waveform = waveform.unsqueeze(0)
                elif waveform.ndim == 2:
                    # 多声道
                    if waveform.shape[1] == 2:
                        waveform = waveform.T  # 转换为(2, samples)
                        if channels == "mono":
                            waveform = waveform.mean(dim=0, keepdim=True)
                    else:
                        # 超过2声道,取前2个
                        waveform = waveform[:2].T
                        if channels == "mono":
                            waveform = waveform.mean(dim=0, keepdim=True)
                
                # 重采样(如果需要)
                if sr != sample_rate:
                    from scipy import signal
                    num_samples = int(waveform.shape[-1] * sample_rate / sr)
                    waveform = torch.from_numpy(
                        signal.resample(waveform.numpy(), num_samples)
                    ).float()
                
                # 归一化
                if normalize:
                    max_val = torch.abs(waveform).max()
                    if max_val > 0:
                        waveform = waveform / max_val
                
                return waveform
                
            except ImportError:
                print("soundfile未安装,尝试使用pydub...")
                try:
                    from pydub import AudioSegment
                    import io
                    
                    # 使用pydub读取
                    audio = AudioSegment.from_file(file_path)
                    
                    # 转换采样率
                    if audio.frame_rate != sample_rate:
                        audio = audio.set_frame_rate(sample_rate)
                    
                    # 转换声道
                    if channels == "mono":
                        audio = audio.set_channels(1)
                    else:
                        audio = audio.set_channels(2)
                    
                    # 转换为numpy数组
                    audio_np = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    
                    # 归一化到[-1, 1]
                    audio_np = audio_np / (2**15)
                    
                    # 转换为torch tensor
                    waveform = torch.from_numpy(audio_np).float()
                    
                    if audio.channels == 2:
                        waveform = waveform.reshape(2, -1)
                    else:
                        waveform = waveform.unsqueeze(0)
                    
                    return waveform
                    
                except ImportError:
                    raise ImportError(
                        "需要安装以下库之一来处理音频: "
                        "librosa, soundfile, 或 pydub. "
                        "请运行: pip install librosa 或 pip install soundfile"
                    )


class AudioInfoExtractor:
    """
    提取音频的基本信息
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }
    
    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("duration", "sample_rate", "channels", "info")
    FUNCTION = "extract_audio_info"
    CATEGORY = "audio/loader"
    
    def extract_audio_info(self, audio: torch.Tensor):
        """
        提取音频信息
        
        Args:
            audio: 音频波形数据, shape为(1, samples)或(2, samples)
            
        Returns:
            duration: 音频时长(秒)
            sample_rate: 采样率(这里返回22050作为默认值)
            channels: 声道数(1或2)
            info: 格式化的信息字符串
        """
        if audio.ndim == 2:
            channels_count = audio.shape[0]
            samples = audio.shape[1]
        elif audio.ndim == 1:
            channels_count = 1
            samples = audio.shape[0]
        else:
            channels_count = 1
            samples = audio.shape[-1]
        
        # 默认采样率(因为我们没有存储采样率信息)
        default_sample_rate = 22050
        duration = samples / default_sample_rate
        
        info_text = (
            f"Duration: {duration:.2f}s | "
            f"Samples: {samples} | "
            f"Channels: {channels_count} | "
            f"Sample Rate: {default_sample_rate}Hz"
        )
        
        return (
            int(duration),
            default_sample_rate,
            channels_count,
            info_text
        )


# 节点映射
NODE_CLASS_MAPPINGS = {
    "AudioLoaderFromURL": AudioLoaderFromURL,
    "AudioInfoExtractor": AudioInfoExtractor,
}

# 节点名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioLoaderFromURL": "Audio Loader from URL (从URL加载音频)",
    "AudioInfoExtractor": "Audio Info Extractor (音频信息提取)",
}
