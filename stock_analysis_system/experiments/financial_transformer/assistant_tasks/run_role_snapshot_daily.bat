@echo off
REM ============================================================
REM  每日角色快照自动刷新
REM  用途：Windows Task Scheduler 定时任务调用
REM  建议触发时间：每天凌晨 2:00
REM  日志输出：assistant_tasks/role_snapshot_cron.log
REM ============================================================

set ROOT=d:\qcr\Stock\stock_analysis_system
set PYTHON=%ROOT%\.venv\Scripts\python.exe
set SCRIPT=%ROOT%\experiments\inverse_rl_market_simulation\scan_market_role_snapshot.py
set CANDIDATE_FILE=%ROOT%\experiments\financial_transformer\results\formal_filtered_v5_e80_bsstop_seed33\latest_predictions.csv
set OUTPUT_DIR=%ROOT%\experiments\inverse_rl_market_simulation\results\market_role_snapshot_daily
set LOGFILE=%ROOT%\experiments\financial_transformer\assistant_tasks\role_snapshot_cron.log

REM 动态获取最近交易日（T-1 或 T-2，避开周末和非交易日）
REM 简单做法：用 Python 生成 end_date
echo [%date% %time%] Starting role snapshot... >> "%LOGFILE%"

%PYTHON% -c "import pandas as pd; from datetime import date, timedelta; d = date.today(); print(d - timedelta(days=1) if d.weekday() < 5 else d - timedelta(days=max(1, d.weekday()-4)))" > "%TEMP%\role_end_date.txt
set /p END_DATE=<"%TEMP%\role_end_date.txt"
del "%TEMP%\role_end_date.txt"

echo [%date% %time%] Using end_date: %END_DATE% >> "%LOGFILE%"

%PYTHON% "%SCRIPT%" ^
    --candidate-file "%CANDIDATE_FILE%" ^
    --candidate-stock-col stock_code ^
    --candidate-score-col signal_score ^
    --candidate-date-col date ^
    --candidate-top-n 54 ^
    --max-trials 20 ^
    --end-date "%END_DATE%" ^
    --output-dir "%OUTPUT_DIR%" >> "%LOGFILE%" 2>&1

echo [%date% %time%] Role snapshot completed with exit code %ERRORLEVEL% >> "%LOGFILE%"
exit /b 0
