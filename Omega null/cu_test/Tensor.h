#pragma once







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
	int[2] size;

private:
	Number[i][j] host_data;
	Number[i][j] device_data;
	bool host_current;
	bool device_current;
	bool synced;
	

};