@echo off
chcp 65001 >nul
title SRT GPT + Voicer Dubber
cd /d "%~dp0"

rem Ищем Python. Пробуем launcher "py", затем "python".
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 "srt_gpt_voicer_dubber_v2.py" %*
    goto :end
)

where python >nul 2>nul
if %errorlevel%==0 (
    python "srt_gpt_voicer_dubber_v2.py" %*
    goto :end
)

echo [ОШИБКА] Python не найден.
echo Установи Python с https://www.python.org/downloads/
echo При установке обязательно отметь галочку "Add Python to PATH".
pause

:end
