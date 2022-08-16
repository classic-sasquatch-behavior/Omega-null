#pragma once
#include"global_manifold.h"





namespace on {

	template<typename NumberType>
	struct Tensor {
	typedef NumberType Number;

	#pragma region data	

		Number* device_data;
		Number* host_data;

		uint num_dims = 0;
		std::vector<uint> spans = { 1, 1, 1, 1 };

		uint& maj_span = spans[0];
		uint& min_span = spans[1];
		uint& cub_span = spans[2];
		uint& hyp_span = spans[3];

		bool synced = false;
		on::host_or_device current = host;

	#pragma endregion

	#pragma region structors

		Tensor() {
			initialize_memory();
			fill_memory(0);
			ready();
		}

		Tensor(std::vector<uint> in_spans, Number constant = 0) {
			int num_dims_in = in_spans.size();
			for (int i = 0; i < num_dims_in; i++) {
				int current_span = in_spans[i];
				if (current_span > 1) {
					spans[i] = in_spans[i];
					num_dims++;
				}
			}

			initialize_memory();
			fill_memory(constant);
			ready();
		}

		//~Tensor() {
		//	cudaFree(device_data);
		//	delete host_data;
		//}

	#pragma endregion

	#pragma region core functions

		void ready() {
			synced = true;
		}

		void sync() {
			if (!synced) {
				switch (current) {
				case on::host: upload(); break;
				case on::device: download(); break;
				default: break; //TODO add error
				}
				ready();
			}
		}

		void desync(on::host_or_device changing) {
			current = changing;
			synced = false;
		}

		//void copy(on::direction input) {
		//	switch (input) {
		//	case on::host_to_device:
		//	case on::device_to_host:
		//	default: break; //TODO add error message
		//	}
		//}

	#pragma endregion

	#pragma region memory functions

		void initialize_memory() {
			cudaMalloc( (void**)&device_data, bytesize());
			host_data = (Number*)malloc(bytesize());
		}

		void fill_memory(Number input) {
			cudaMemset(device_data, input, bytesize());
			std::fill_n(host_data, num_elements(), input);
		}

		void upload() {
			cudaMemcpy(device_data, host_data, bytesize(), cudaMemcpyHostToDevice);
		}

		void download() {
			cudaMemcpy(host_data, device_data, bytesize(), cudaMemcpyDeviceToHost);
		}

	#pragma endregion

	#pragma region fetching functions

		//get data
		__host__ __device__ Number& operator ()(int maj, int min = 0, int cub = 0, int hyp = 0) {
			#ifdef __CUDA_ARCH__			
			return device_data[(maj * min_span) + min];
			#else
			return host_data[(maj * min_span) + min];
			#endif
		}

		int num_elements() { return maj_span * min_span; }
		int bytesize() { return (num_elements() * sizeof(NumberType)); }

	#pragma endregion


	#pragma region interop
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



		//doesn't work
		operator af::array() { af::array temp(maj_span, min_span, const_cast<const uchar*>(device_data), afDevice); return temp; }
	#pragma endregion

	};


	#ifdef ON_USE_RANDOM
	template<typename Number>
	Tensor<Number>& randu(uint maj_span, uint min_span);


	Tensor<uchar> rand_ones_and_zeroes(uint maj_span, uint min_span);
	#endif

}