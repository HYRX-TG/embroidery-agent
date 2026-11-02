# 知绣坊 · 数据备份脚本（客户机日常自动执行）
# 用法：powershell -ExecutionPolicy Bypass -File backup.ps1 [项目根目录]
# 计划任务示例（每天 03:00）：
#   schtasks /create /tn "ZhixiufangBackup" /tr "powershell -ExecutionPolicy Bypass -File D:\embroidery_agent\backup.ps1 D:\embroidery_agent" /sc daily /st 03:00 /f

param([string]$ProjectRoot = "")

if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$ErrorActionPreference = "Stop"
$BackupRoot = Join-Path $ProjectRoot "backups"
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Target = Join-Path $BackupRoot ("data-" + $Stamp)
New-Item -ItemType Directory -Force -Path $Target | Out-Null

$Copied = 0

# 核心数据（会话/工单/渠道绑定/黄金测试集）
$Files = @("data\embroidery.db", "data\golden_test.jsonl")
foreach ($rel in $Files) {
    $f = Join-Path $ProjectRoot $rel
    if (Test-Path $f) {
        Copy-Item $f $Target -Force
        $Copied = $Copied + 1
    }
}

# 知识库源文件（客户可自行维护的 md 全部备份）
$Dirs = @("data\knowledge", "data\internal")
foreach ($rel in $Dirs) {
    $dir = Join-Path $ProjectRoot $rel
    if (Test-Path $dir) {
        $leaf = Split-Path $rel -Leaf
        $dest = Join-Path $Target $leaf
        Copy-Item $dir $dest -Recurse -Force
    }
}

# 保留最近 14 份，清理更早的
$Old = Get-ChildItem $BackupRoot -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "data-*" } |
    Sort-Object Name -Descending |
    Select-Object -Skip 14
foreach ($d in $Old) {
    Remove-Item $d.FullName -Recurse -Force
}

$Size = (Get-ChildItem $Target -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Output ("[OK] 备份完成: " + $Target + "  (" + [math]::Round($Size / 1KB, 1) + " KB, 核心文件 " + $Copied + " 个)")
