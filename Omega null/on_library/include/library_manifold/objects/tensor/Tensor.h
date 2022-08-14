#pragma once
#include"global_manifold.h"





namespace on {

	template<typename NumberType>
	struct Tensor {
		typedef NumberType Number;
		public:
		Number* device_data;
		Number* host_data;

		uint num_dims = 0;

		uint maj_span = 1;
		uint min_span = 1;
		uint cub_span = 1;
		uint hyp_span = 1;

		uint* spans[4] = { &maj_span, &min_span, &cub_span, &hyp_span };

		private:
		bool synced = false;
		on::host_or_device current = 0;

		void ready() {
			synced = true;
		}

		void sync() {
			if (!synced) {
				switch (current) {
				case on::host: copy(on::host_to_device); break;
				case on::device: copy(on::device_to_host); break;
				default: break; //TODO add error
				}
				synced = true; //TODO overload ++ for bools
			}
		}

		void desync(on::host_or_device change) {
			current = change;
			synced = false;
		}

		public:
		Tensor() { 
			initialize_memory(); 
			fill_memory(0); 
			ready();
		}

		Tensor(uint* in_spans[], Number constant = 0){
			num_dims = sizeof(in_spans)/sizeof(in_spans[0]);
			for (int i = 0; i < num_dims; i++) {
				spans[i] = in_spans[i];
			}

			initialize_memory();
			fill_memory(_constant);
			ready();
		}

		public:
		int num_elements() { return maj_span * min_span; }
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

		void copy(on::direction input) {
			switch (input) {
			case on::host_to_device:
			case on::device_to_host:
			default: break; //TODO add error message
			}
		}

	#pragma region interop
		public:
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