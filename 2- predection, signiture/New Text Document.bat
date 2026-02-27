@echo off
REM EnergyPlus simulation script

REM === 1. مسیر نصب EnergyPlus (در صورت نیاز تغییر دهید) ===
set EPLUS_EXE="C:\EnergyPlusV9-6-0\energyplus.exe"

REM === 2. نام فایل‌ها ===
set IDF_FILE="Res_flat1.idf"
set EPW_FILE="Torino_IT-hour.epw"

REM === 3. اجرای EnergyPlus ===
%EPLUS_EXE% -w %EPW_FILE% -r %IDF_FILE%

REM === 4. بررسی موفقیت اجرا ===
if exist eplusout.csv (
    echo Simulation completed successfully.
    echo Output file: eplusout.csv
) else (
    echo Simulation failed or output file not found!
)
pause
