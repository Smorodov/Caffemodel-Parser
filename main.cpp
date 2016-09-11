
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include "caffe.pb.h"
#include <fstream>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

// for "CreateDirectory"
#include <Windows.h>
// for "_open" function in VS, use "open" in linux instead this
#include <io.h>
#include <fcntl.h>

using namespace google::protobuf::io;
using namespace std;
using namespace caffe;

void dumpFilters(LayerParameter& lp, string layer_name, string result_folder)
{
	for (int i = 0; i < lp.blobs_size(); ++i)
	{
		string fname = result_folder + "/" + layer_name + "_b" + std::to_string(i) + ".bin";

		int nDims = lp.blobs(i).shape().dim_size();
		int blob_sz = lp.blobs(i).data_size();
		cout << "blob_sz=" << blob_sz << endl;
		cout << "fname=" << fname << endl;
		if (nDims > 0 && blob_sz > 0)
		{
			ofstream ofs(fname, std::ios::out | std::ios::binary);
			for (int n = 0; n < blob_sz; ++n)
			{
				float d = lp.blobs(i).data(n);
				ofs.write((char*)&d, sizeof(float));
			}
			ofs.close();
		}
	}
}

void printBlobsInfo(LayerParameter& lp)
{
	for (int i = 0; i < lp.blobs_size(); ++i)
	{
		cout << "Blob[" << i << "] size: ";
		if (lp.blobs(i).shape().dim_size() > 0)
		{
			cout << "[";
		}
		for (int j = 0; j < lp.blobs(i).shape().dim_size(); ++j)
		{
			if (j != lp.blobs(i).shape().dim_size() - 1)
			{
				cout << lp.blobs(i).shape().dim(j) << "x";
			}
			else
			{
				cout << lp.blobs(i).shape().dim(j) << "]" << endl;
			}
		}
	}
}

int main(int argc, char* argv[])
{
	string modelName = "lenet_iter_10000";
	CreateDirectory(modelName.c_str(), NULL);
	caffe::NetParameter msg;
	int fd = _open((modelName + ".caffemodel").c_str(), O_RDONLY | O_BINARY);
	ZeroCopyInputStream* raw_input = new FileInputStream(fd);
	CodedInputStream* coded_input = new CodedInputStream(raw_input);
	// Remove protobuf limits
	coded_input->SetTotalBytesLimit(INT_MAX, INT_MAX);

	if (!msg.ParseFromCodedStream(coded_input))
	{
		cerr << "Failed to parse." << endl;
		return -1;
	}
	cout << "Layers number: " << msg.layer_size() << endl;
	::google::protobuf::RepeatedPtrField< LayerParameter >* layer = msg.mutable_layer();
	::google::protobuf::RepeatedPtrField< LayerParameter >::iterator it = layer->begin();
	for (; it != layer->end(); ++it)
	{
		cout << "-------------" << endl;
		cout << "Layer name:" << it->name() << endl;
		cout << "Layer type:" << it->type() << endl;
		printBlobsInfo(*it);
		dumpFilters(*it, it->name(), modelName);
	}
	delete raw_input;
	delete coded_input;
	cout << " Done. Press any key" << endl;
	getchar();
	return 0;
}