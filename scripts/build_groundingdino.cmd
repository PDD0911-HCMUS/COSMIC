@echo off
setlocal
set "BUILD_PYTHON=%~1"
if not defined BUILD_PYTHON set "BUILD_PYTHON=python"
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
set "TORCH_CUDA_ARCH_LIST=8.6"
set "MAX_JOBS=2"
pushd "%~dp0..\data\OvSGTR\GroundingDINO"
"%BUILD_PYTHON%" setup.py build_ext --inplace
set "BUILD_RESULT=%ERRORLEVEL%"
popd
exit /b %BUILD_RESULT%
