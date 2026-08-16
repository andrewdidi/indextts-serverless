"""
RunPod Serverless 入口（控制台 Git 扫描认此文件 + runpod.serverless.start）。

Index-TTS_Serverless · IndexTTS-2.5（默认）零样本音色克隆 TTS
"""

import runpod

from handler import handler

runpod.serverless.start({"handler": handler})
