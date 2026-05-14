@echo off
cd /d D:\ai-projects\IndianModelsEval
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
".venv\Scripts\pythonw.exe" scripts\run_eval_cli.py --model sarvam-translate --batch-size 32 > logs\sarvam-full.log 2>&1
