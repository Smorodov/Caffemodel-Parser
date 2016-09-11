0. git clone --recursive https://github.com/Smorodov/Caffemodel-Parser.git
1. Run CMake and create VS solution.
2. Build protoc compiler.
3. Run generate_caffe_pb_cc.bat (if caffe.pb.cc and caffe.pb.h does not match your protobuf version, or absent)
It will create caffe.pb.cc and caffe.pb.h using caffe.proto file, that you can find in your caffe sources if it absent, or you want to use your version of caffe protocol definitions.
4. Build project.