@echo off
chcp 65001 >nul
echo ========================================
echo    SCOPE 项目上传到 GitHub
echo ========================================
echo.

cd /d "%~dp0"

REM 检查 git 是否安装
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Git，请先安装 Git: https://git-scm.com/download/win
    pause
    exit /b 1
)

REM 设置 GitHub 仓库地址
set GITHUB_REPO=https://github.com/sjy-as/SCOPE.git
set GITHUB_BRANCH=main

REM 获取用户输入的 GitHub Token
echo 请输入你的 GitHub Personal Access Token:
echo (访问 https://github.com/settings/tokens 生成，需要 repo 权限)
echo.
set /p GITHUB_TOKEN=Token: 

if "%GITHUB_TOKEN%"=="" (
    echo [错误] Token 不能为空
    pause
    exit /b 1
)

echo.
echo [1/5] 检查 Git 仓库状态...
if not exist ".git" (
    echo 初始化 Git 仓库...
    git init
    git branch -M %GITHUB_BRANCH%
) else (
    echo Git 仓库已存在
)

echo.
echo [2/5] 设置远程仓库...
git remote remove origin >nul 2>&1
git remote add origin https://%GITHUB_TOKEN%@github.com/sjy-as/SCOPE.git
echo 远程仓库已设置

echo.
echo [3/5] 检查 .gitignore 文件...
if not exist ".gitignore" (
    echo 创建 .gitignore 文件...
    (
        echo # Python
        echo __pycache__/
        echo *.py[cod]
        echo *$py.class
        echo .venv/
        echo venv/
        echo env/
        echo.
        echo # IDE
        echo .vscode/
        echo .idea/
        echo *.ipynb_checkpoints/
        echo.
        echo # OS
        echo .DS_Store
        echo Thumbs.db
        echo.
        echo # Secrets
        echo .env
        echo .env.*
        echo.
        echo # Large data files
        echo nba-datalake.wiki-documents.json
        echo.
        echo # Logs
        echo *.log
    ) > .gitignore
)

echo.
echo [4/5] 添加文件到暂存区...
git add -A

echo.
echo 当前暂存的文件:
git status --short

echo.
echo [5/5] 提交更改...
set /p COMMIT_MSG=请输入提交信息 (直接回车使用默认): 
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Initial commit: SCOPE project

git commit -m "%COMMIT_MSG%"

echo.
echo ========================================
echo    正在推送到 GitHub...
echo ========================================
echo.

git push -u origin %GITHUB_BRANCH%

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo    上传成功！
    echo.
    echo    仓库地址: https://github.com/sjy-as/SCOPE
    echo ========================================
) else (
    echo.
    echo [错误] 上传失败，请检查:
    echo   1. Token 是否有 repo 权限
    echo   2. 网络连接是否正常
    echo   3. 仓库地址是否正确
)

echo.
pause
