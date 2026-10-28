# 角色快照定时任务设置说明

## 任务 1：Filter 试验线角色快照（54 只，每天 10 分钟）

### Windows Task Scheduler 注册命令（管理员 PowerShell）：

```powershell
$action = New-ScheduledTaskAction -Execute "d:\qcr\Stock\stock_analysis_system\experiments\financial_transformer\assistant_tasks\run_role_snapshot_daily.bat"
$trigger = New-ScheduledTaskTrigger -Daily -At 1:30AM
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
Register-ScheduledTask -TaskName "StockRoleSnapshot_Filter54" -Action $action -Trigger $trigger -Principal $principal -Description "每日凌晨扫描Filter 54只候选池角色"
```

## 任务 2：全市场角色快照（TOP 200 按成交额，每天 15-20 分钟）

### Windows Task Scheduler 注册命令（管理员 PowerShell）：

```powershell
$action = New-ScheduledTaskAction -Execute "d:\qcr\Stock\stock_analysis_system\experiments\financial_transformer\assistant_tasks\run_role_snapshot_fullmarket.bat"
$trigger = New-ScheduledTaskTrigger -Daily -At 2:00AM
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
Register-ScheduledTask -TaskName "StockRoleSnapshot_FullMarket200" -Action $action -Trigger $trigger -Principal $principal -Description "每日凌晨扫描全市场成交额TOP200角色快照"
```

## 全市场扫描说明

- **扫描范围**：按当日成交额排序的前 200 只最活跃股票
- **耗时**：约 15-20 分钟（200 只 × 5 trials）
- **全市场 5768 只全部扫描**：不推荐一次性跑（约 17 小时）。如需更大覆盖，可改为每天扫不同批次（比如按成交额分段）。
- **调参**：修改 `run_role_snapshot_fullmarket.bat` 中的 `TOPN` 和 `MAX_TRIALS`

## 日志位置

| 任务 | 日志 |
|---|---|
| Filter 54 只 | `assistant_tasks/role_snapshot_cron.log` |
| 全市场 Top 200 | `assistant_tasks/role_snapshot_fullmarket_cron.log` |

## 输出文件

```
experiments/inverse_rl_market_simulation/results/market_role_snapshot_daily/
  ├── market_role_snapshot.csv              ← 原始快照
  ├── market_role_snapshot_with_signals.csv ← 与信号合并（signal_report 读这个）
  ├── market_role_snapshot_failures.csv
  ├── market_role_snapshot.md
  └── market_role_snapshot.json
```

## 手动运行

```cmd
rem Filter 54 只版
d:\qcr\Stock\stock_analysis_system\experiments\financial_transformer\assistant_tasks\run_role_snapshot_daily.bat

rem 全市场 TOP 200 版
d:\qcr\Stock\stock_analysis_system\experiments\financial_transformer\assistant_tasks\run_role_snapshot_fullmarket.bat
```
