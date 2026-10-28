@echo off
REM ============================================================
REM  每日全市场角色快照
REM  扫描全部股票，约10分钟
REM  建议触发：每天凌晨 2:00
REM  日志：assistant_tasks/role_snapshot_fullmarket_cron.log
REM ============================================================

set ROOT=d:\qcr\Stock\stock_analysis_system
set PYTHON=%ROOT%\.venv\Scripts\python.exe
set SCRIPT=%ROOT%\experiments\inverse_rl_market_simulation\scan_market_role_snapshot.py
set OUTPUT_DIR=%ROOT%\experiments\inverse_rl_market_simulation\results\market_role_snapshot_daily_all
set LOGFILE=%ROOT%\experiments\financial_transformer\assistant_tasks\role_snapshot_fullmarket_cron.log

echo ============================================================ >> "%LOGFILE%"
echo [%date% %time%] === Full-market role snapshot === >> "%LOGFILE%"

REM 动态获取最近交易日
%PYTHON% -c "from datetime import date, timedelta; d = date.today(); print(d - timedelta(days=1) if d.weekday() ^< 5 else d - timedelta(days=max(1, d.weekday()-4)))" > "%TEMP%\role_end_date.txt
set /p END_DATE=<"%TEMP%\role_end_date.txt"
del "%TEMP%\role_end_date.txt"
echo [%date% %time%] END_DATE=%END_DATE% >> "%LOGFILE%"

echo [%date% %time%] Scanning all stocks... >> "%LOGFILE%"
%PYTHON% "%SCRIPT%" --max-trials 3 --end-date "%END_DATE%" --output-dir "%OUTPUT_DIR%" >> "%LOGFILE%" 2>&1

echo [%date% %time%] Completed with exit code %ERRORLEVEL% >> "%LOGFILE%"
exit /b 0
