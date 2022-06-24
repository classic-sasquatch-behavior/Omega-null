#pragma once
#include"../headers/host_headers.h"






template<typename Number, int i, int j>
class Tensor {
public:
	Tensor();
	~Tensor();
	void sync();
	void upload();
	void download();
	int rows;
	int cols;
	int x_dim;
	int y_dim;
	int dims;
	int size[2];

private:
	Number host_data[i][j];
	Number device_data[i][j];
	bool host_current;
	bool device_current;
	bool synced;
	

};