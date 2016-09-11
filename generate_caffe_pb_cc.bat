copy caffe.proto ./build/Release
cd ./build/Release
protoc.exe caffe.proto --cpp_out=../../
pause