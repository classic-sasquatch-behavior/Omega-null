#pragma once
#include<manifold.h>






namespace on {
	template<typename NumberType>
	struct Tensor {
	typedef NumberType Number;
	public:
		Number* device_data;
		Number* host_data;

		int maj_span;
		int min_span;

		Tensor() { maj_span = 0; min_span = 0; initialize_memory(); fill_memory(0); sync(); }
		Tensor(int _maj_span, int _min_span, Number _constant) {
			maj_span = _maj_span;
			min_span = _min_span;
			initialize_memory();
			fill_memory(_constant);
			sync();
		}

		inline int num_elements() { return maj_span * min_span; }
		inline int bytesize() { return (num_elements() * sizeof(NumberType)); }

		//the host side of these next two functions are kinda messed up
		void initialize_memory() {
			cudaMalloc(&device_data, bytesize());
		}

		void fill_memory(Number input) {
			cudaMemset(device_data, input, bytesize());
			//host_data = { new Number[enumsize()] = { input } }; //whut
		}

		void upload() {
			cudaMemcpy(device_data, host_data, bytesize(), cudaMemcpyHostToDevice);
		}

		void download() {
			cudaMemcpy(host_data, device_data, bytesize(), cudaMemcpyDeviceToHost);
		}

		__host__ __device__ void operator=(Tensor input) {
			maj_span = input.maj_span;
			min_span = input.min_span;
			host_data = input.host_data;
			device_data = input.device_data;
		}

		//doesn't work
		__host__ __device__ void operator=(af::array input) {
			maj_span = input.dims(0);
			min_span = input.dims(1);
			//host_data;
			device_data = input.device<Number>();
		}

		__device__ Number& operator ()(int maj, int min) { return device_data[(maj * min_span) + min]; }
		//__host__ Number& operator ()(int maj, int min) { return host_data[(maj * min_span) + min]; }

		//~Tensor() {
		//	cudaFree(device_data);
		//	//delete host_data;
		//}

		//attempt at interoperability with array
		operator af::array() { af::array temp(maj_span, min_span, const_cast<const uchar*>(device_data), afDevice); return temp; }

	private:
		bool synced = false;



		inline void sync() {
			if (synced) { return; }
			synced = true;
		}

		inline void desync() {
			synced = false;
		}

	};


	template<typename Number>
	Tensor<Number>& randu(uint maj_span, uint min_span);


	Tensor<uchar> rand_ones_and_zeroes(uint maj_span, uint min_span);


}