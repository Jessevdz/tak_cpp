rm -r build
rm C:\Users\Jesse\Projects\tak_cpp\tak_cpp.cp39-win_amd64.pyd
rm C:\Users\Jesse\Projects\tak_cpp\tak_cpp.lib
rm C:\Users\Jesse\Projects\tak_cpp\tak_cpp.exp
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=C:\Users\Jesse\Projects\tak_cpp\lib\libtorch_debug .. -Dpybind11_DIR=C:\Users\Jesse\Projects\tak_cpp\.venv\Lib\site-packages\pybind11\share\cmake\pybind11 -DPYTHON_EXECUTABLE:FILEPATH=C:\Users\Jesse\Projects\tak_cpp\.venv\Scripts\python.exe
cmake --build . --config Debug

cp C:\Users\Jesse\Projects\tak_cpp\build\Debug\tak_cpp.cp39-win_amd64.pyd C:\Users\Jesse\Projects\tak_cpp\tak_cpp.cp39-win_amd64.pyd
cp C:\Users\Jesse\Projects\tak_cpp\build\Debug\tak_cpp.lib C:\Users\Jesse\Projects\tak_cpp\tak_cpp.lib
cp C:\Users\Jesse\Projects\tak_cpp\build\Debug\tak_cpp.exp C:\Users\Jesse\Projects\tak_cpp\tak_cpp.exp