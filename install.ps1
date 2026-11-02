# 知绣坊 · 客户机一键安装/升级脚本（Windows）
# 用法：右键"以管理员身份运行"，或：powershell -ExecutionPolicy Bypass -File install.ps1
#
# 做什么：
#   1. 检查 Python / Ollama / 模型 是否就绪（缺什么提示装什么，不静默安装）
#   2. 安装/更新项目依赖
#   3. 构建/校验知识库索引
#   4. 用 NSSM 把服务注册为开机自启（nssm.exe 需与本脚本同目录，或已加入 PATH）
#   5. 注册每日 03:00 数据备份计划任务
#   6. 启动服务

param([string]$ProjectRoot = "")

if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}
Set-Location $ProjectRoot

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  知绣坊 · 客户机安装/升级" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

$fail = $false

# ---------- 1. Python ----------
$py = $null
foreach ($cand in @("python", "py")) {
    $cmd = Get-Command $cand -ErrorAction SilentlyContinue
    if ($cmd) { $py = $cmd.Source; break }
}
if (-not $py) {
    Write-Host "[缺失] 未找到 Python。请到 https://www.python.org/downloads/ 安装 3.10+（安装时勾选 Add to PATH）" -ForegroundColor Red
    $fail = $true
} else {
    Write-Host "[OK] Python: $py" -ForegroundColor Green
}

# ---------- 2. Ollama 与模型 ----------
$ollamaOk = $false
try {
    $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3
    $names = @($tags.models | ForEach-Object { $_.name })
    foreach ($need in @("qwen2.5:7b-instruct", "bge-m3")) {
        if ($names -contains $need) {
            Write-Host "[OK] 模型: $need" -ForegroundColor Green
        } else {
            Write-Host "[缺失] 模型 $need 未拉取，请运行: ollama pull $need" -ForegroundColor Red
            $fail = $true
        }
    }
    $ollamaOk = $true
} catch {
    Write-Host "[缺失] Ollama 未运行。请先安装并启动 Ollama（https://ollama.com/download）" -ForegroundColor Red
    $fail = $true
}

if ($fail) {
    Write-Host ""
    Write-Host "环境未就绪，请先补齐上述缺失项后重试。" -ForegroundColor Yellow
    exit 1
}

# ---------- 3. 依赖 ----------
if ($py) {
    Write-Host "[步骤] 安装/更新依赖（首次需几分钟）…" -ForegroundColor Cyan
    & $py -m pip install --upgrade pip 2>&1 | Out-Null
    & $py -m pip install -e ".[api,retrieval]" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[错误] 依赖安装失败，请检查网络或执行: python -m pip install -e '.[api,retrieval]'" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] 依赖安装完成" -ForegroundColor Green
}

# ---------- 4. 知识库索引 ----------
$env:PYTHONPATH = "src"
if ($py) {
    & $py scripts\ensure_kb.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[警告] 知识库索引校验未通过，服务启动时会重试" -ForegroundColor Yellow
    } else {
        Write-Host "[OK] 知识库索引就绪" -ForegroundColor Green
    }
}

# ---------- 5. 注册 Windows 服务（开机自启） ----------
$nssm = Get-Command nssm -ErrorAction SilentlyContinue
if (-not $nssm) { $nssm = Get-Item (Join-Path $ProjectRoot "nssm.exe") -ErrorAction SilentlyContinue }
$serviceName = "ZhixiufangAgent"
if ($nssm) {
    $nssmPath = $nssm.Source
    & $nssmPath stop $serviceName 2>&1 | Out-Null
    & $nssmPath remove $serviceName confirm 2>&1 | Out-Null
    & $nssmPath install $serviceName $py "-m uvicorn embroidery_agent.api.server:app --host 0.0.0.0 --port 8000" | Out-Null
    & $nssmPath set $serviceName AppDirectory $ProjectRoot | Out-Null
    & $nssmPath set $serviceName AppEnvironmentExtra "PYTHONPATH=src" | Out-Null
    & $nssmPath set $serviceName Start SERVICE_AUTO_START | Out-Null
    & $nssmPath set $serviceName AppStdout (Join-Path $ProjectRoot "logs\service.out.log") | Out-Null
    & $nssmPath set $serviceName AppStderr (Join-Path $ProjectRoot "logs\service.err.log") | Out-Null
    & $nssmPath start $serviceName | Out-Null
    Write-Host "[OK] 服务已注册并启动（开机自启）：$serviceName" -ForegroundColor Green
} else {
    Write-Host "[提示] 未找到 nssm.exe，跳过服务注册。" -ForegroundColor Yellow
    Write-Host "        开机自启方案一：把 nssm.exe 放到项目目录后重跑本脚本" -ForegroundColor Yellow
    Write-Host "        开机自启方案二：手动添加启动项 → shell:startup → 放 start-后台.vbs" -ForegroundColor Yellow
    & $py -m uvicorn embroidery_agent.api.server:app --host 0.0.0.0 --port 8000 2>&1 | Out-Null
    Write-Host "[OK] 服务已前台启动（临时）" -ForegroundColor Green
}

# ---------- 6. 每日备份计划任务 ----------
$backupTask = "ZhixiufangBackup"
schtasks /Create /TN $backupTask /TR "powershell -ExecutionPolicy Bypass -File `"$(Join-Path $ProjectRoot 'backup.ps1')`" `"$ProjectRoot`"" /SC DAILY /ST 03:00 /F | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] 每日 03:00 自动备份已注册" -ForegroundColor Green
} else {
    Write-Host "[警告] 备份计划任务注册失败（管理员权限？）" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "部署完成。浏览器访问: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "（对接到微信/直播等渠道需公网地址，见 README 渠道接入章节）" -ForegroundColor DarkGray
