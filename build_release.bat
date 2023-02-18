rm -r build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=C:\Users\Jesse\Projects\tak_cpp\libtorch_release ..
cmake --build . --config Release