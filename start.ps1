# 知绣坊 · 一键启动脚本（Windows PowerShell）
# 用法：右键 → 使用 PowerShell 运行；或在此目录下执行  .\start.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  知绣坊 · 刺绣企业本地智能客服" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# 1. 检查 Ollama
$ollamaRunning = $false
try {
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3
    $ollamaRunning = $true
} catch { }

if (-not $ollamaRunning) {
    Write-Host "[警告] Ollama 服务未运行！" -ForegroundColor Yellow
    Write-Host "  - 请先启动 Ollama（开始菜单搜索 Ollama，或运行 ollama serve）" -ForegroundColor Yellow
    Write-Host "  - 再运行: ollama pull qwen2.5:7b-instruct  和  ollama pull bge-m3" -ForegroundColor Yellow
    Write-Host "  系统将降级为「知识库直接引用」模式运行（检索仍可用）。" -ForegroundColor Yellow
} else {
    $modelNames = @($resp.models | ForEach-Object { $_.name })
    if ($modelNames -contains "qwen2.5:7b-instruct") {
        Write-Host "[OK] Ollama 运行中，已检测到 qwen2.5:7b-instruct" -ForegroundColor Green
    } else {
        Write-Host "[警告] 未检测到 qwen2.5:7b-instruct，可运行: ollama pull qwen2.5:7b-instruct" -ForegroundColor Yellow
    }
    if ($modelNames -contains "bge-m3") {
        Write-Host "[OK] 已检测到 bge-m3 embedding 模型" -ForegroundColor Green
    } else {
        Write-Host "[警告] 未检测到 bge-m3，可运行: ollama pull bge-m3" -ForegroundColor Yellow
    }
}

# 2. 确保知识库索引为最新（源文件变更自动重建）
$env:PYTHONPATH = "src"
Write-Host "[检查] 知识库索引（源文件变更自动重建）…" -ForegroundColor Cyan
python scripts/ensure_kb.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[错误] 知识库检查失败，请检查 data/knowledge 与 data/internal 目录" -ForegroundColor Red
    exit 1
}

# 3. 启动服务
Write-Host ""
Write-Host "启动服务中… 浏览器访问: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "（按 Ctrl+C 停止服务）" -ForegroundColor DarkGray
uvicorn embroidery_agent.api.server:app --host 127.0.0.1 --port 8000
