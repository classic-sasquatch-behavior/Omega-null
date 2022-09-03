#pragma once
#include"global_manifold.h"




namespace on {

	template<typename Number>
	struct Tensor {

	#pragma region data	

		Number* device_data = nullptr;
		Number* host_data = nullptr;

		uint num_dims = 0;

		uint* host_spans = nullptr;
		uint* device_spans = nullptr;

		__device__ __host__ uint maj_span(){
			#ifdef __CUDA_ARCH__
				return device_spans[0];
			#else
				return host_spans[0];
			#endif
		}
		__device__ __host__ uint min_span() {
			#ifdef __CUDA_ARCH__
				return device_spans[1];
			#else
				return host_spans[1];
			#endif
		}
		__device__ __host__ uint cub_span() {
			#ifdef __CUDA_ARCH__
				return device_spans[2];
			#else
				return host_spans[2];
			#endif
		}
		__device__ __host__ uint hyp_span() {
			#ifdef __CUDA_ARCH__
				return device_spans[3];
			#else
				return host_spans[3];
			#endif
		}

		bool synced = false;
		on::host_or_device current = on::host;

	#pragma endregion

	#pragma region get data

		//get data
		__host__ __device__ Number& operator ()(int maj, int min = 0, int cub = 0, int hyp = 0) {
			#ifdef __CUDA_ARCH__			
				return device_data[(((((maj * min_span()) + min) * cub_span()) + cub) * hyp_span()) + hyp];
			#else
				return host_data[(((((maj * min_span()) + min) * cub_span()) + cub) * hyp_span()) + hyp];
			#endif
		}

		int num_elements() { return maj_span() * min_span() * cub_span() * hyp_span(); }
		int bytesize() { return (num_elements() * sizeof(Number)); }

#pragma endregion

	#pragma region structors



		Tensor(Number constant = 0) {
			initialize_spans();
			initialize_memory();
			fill_memory(constant);
			ready();
		}

		Tensor(std::vector<uint> in_spans, Number constant = 0) {
			initialize_spans();
			int num_dims_in = in_spans.size();
			num_dims = 0;
			for (int i = 0; i < num_dims_in; i++) {
				int current_span = in_spans[i];
				if (current_span > 1) {
					host_spans[i] = in_spans[i];
					num_dims++;
				}
			}
			initialize_memory();
			fill_memory(constant);
			ready();
		}

		~Tensor() {
			cudaFree(device_data);
			delete host_data;
			//free(host_data); //should be free, because I used malloc in initialize memory... right? 
							 //evidently not, because I still get that heap error. that or theres another problem.
							 //the fact that I'm just cutting and running in the main function probbaly doesnt help.
		}

	#pragma endregion

	#pragma region core functions

		//need to find a way to make sync automatic when tensor is passed to CUDA kernel. I would prefer to do this without creating another type, which is one solution.


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

	#pragma endregion

	#pragma region memory functions

		#pragma region initialize memory
		void initialize_spans() {
			host_spans = new uint[4];
			std::fill_n(host_spans, 4, 1);
			cudaMalloc((void**)&device_spans, 4 * sizeof(uint));
			cudaMemcpy(device_spans, host_spans, 4 * sizeof(uint), cudaMemcpyHostToDevice);
		}

		void initialize_memory() {
			initialize_host_memory();
			initialize_device_memory();
		}

		void initialize_host_memory() {
			host_data = new Number[num_elements()];

		}

		void initialize_device_memory() {
			cudaMalloc((void**)&device_data, bytesize());

		}
		#pragma endregion

		#pragma region fill memory

		void fill_memory(Number input) {
			fill_host_memory(input);
			fill_device_memory(input);
		}

		void fill_host_memory(Number input) {
			std::fill_n(host_data, num_elements(), input);
		}

		void fill_device_memory(Number input) {
			cudaMemset(device_data, input, bytesize());
			cudaMemset(device_spans, 1, 4*sizeof(uint));
		}

		#pragma endregion

		#pragma region transfer memory

		//bytesize will always be accurate as long as the meta data is accurate, but can cuda memcpy allocate space on its own, or should I be managing 
		//the memory completely manually? (i.e. freeing and reallocating it every time).
		//CUDA has gotten pretty good about stuff like that recently so for now I wont worry about it unless it becomes a problem. If you start getting
		//I guess device memory leaks, start here by controlling the memory more carefully.
		void upload() {
			cudaMemcpy(device_data, host_data, bytesize(), cudaMemcpyHostToDevice);
			cudaMemcpy(device_spans, host_spans, 4*sizeof(uint), cudaMemcpyHostToDevice);
		}

		void download() {
			cudaMemcpy(host_data, device_data, bytesize(), cudaMemcpyDeviceToHost);
			cudaMemcpy(host_spans, device_spans, 4*sizeof(uint), cudaMemcpyDeviceToHost);
		}

		#pragma endregion

	#pragma endregion



	#pragma region interop

		#pragma region omega null

		Tensor(const Tensor& input) {
			initialize_spans();
			initialize_memory();
			On_Copy(host_spans, input.host_spans, 4);
			cudaMemcpy(device_spans, host_spans, 4 * sizeof(uint), cudaMemcpyHostToDevice);
			num_dims = input.num_dims;
			host_data = input.host_data;
			device_data = input.device_data;
			ready();
		}

		void operator=(const Tensor& input){
			On_Copy(host_spans, input.host_spans, 4);
			cudaMemcpy(device_spans, host_spans, 4 * sizeof(uint), cudaMemcpyHostToDevice);
			num_dims = input.num_dims;
			host_data = input.host_data;
			device_data = input.device_data;
		}


		#pragma endregion

		#pragma region std

			//from vector
			void operator=(std::vector<Number> input) {
				desync(host);
				num_dims = 1;
				host_spans[0] = input.size();

				//careful with this one, something about how it handles memory could be throwing things off with the destructor
				std::copy(input.begin(), input.end(), host_data);

				sync();
			}

			//to vector
			operator std::vector<Number>() { return std::vector<Number>(host_data, host_data + num_elements()); }

			//to pointer
			__host__ __device__ operator Number* () {
				#ifdef __CUDA_ARCH__
				return device_data;
				#else
				return host_data;
				#endif
			}

			//from numerical type

			__host__ __device__ operator Number () {
				#ifdef __CUDA_ARCH__
				return device_data[0];
				#else 
				return host_data[0];
				#endif
			}

			//to numerical type - maybe this could take the sum of the array? would therefore be consistent with the use case I want it for (casting a 0 dim Tensor to a number)

			//comparison to numerical type
			bool operator ==(Number compare){
				Number self = host_data[0];
				return compare == self;
			}


		#pragma endregion

		#pragma region ArrayFire

			//from array
			void operator=(af::array& input) {
				
				desync(host);
				num_dims = input.numdims();
				for (int i = 0; i < num_dims; i++) {
					host_spans[i] = input.dims(i);
				}
				sync();

				desync(device);
				cudaMemcpy((void*)device_data, (void*)input.device<Number>(), bytesize(), cudaMemcpyDeviceToDevice);
				input.unlock(); //probably a sloppy way to do this, but oh well
				sync();
			}

			//to array
			operator af::array() { return af::array((dim_t)maj_span(), (dim_t)min_span(), host_data); }

		#pragma endregion

		#pragma region OpenCV

			//from Mat to Tensor
			void operator=(cv::Mat input) {
				num_dims = input.dims;
				bool has_channels = (input.channels() > 1);
				num_dims += has_channels;

				desync(host);

				host_spans[0] = input.rows;
				host_spans[1] = input.cols;
				host_spans[2] = input.channels();

				host_data = (Number*)input.data;
				sync();
			}

			//from Tensor to Mat
			operator cv::Mat() {
				return cv::Mat(host_spans[0], host_spans[1], cv::DataType<Number>::type, host_data);
			}

			#ifdef GPUMAT_TO_TENSOR_FIXED
			//from GpuMat to Tensor
			void operator=(cv::cuda::GpuMat input) {
				cv::Mat temp;
				input.download(temp);

				num_dims = temp.dims;
				spans[0] = temp.rows;
				spans[1] = temp.cols;

				desync(host);
				std::copy((Number*)temp.data, (Number *)temp.data[temp.rows * temp.cols], host_data);
				sync();
			}

			cv::cuda::GpuMat make_gpumat() {
				cv::Mat temp;
				temp = *this;
				cv::cuda::GpuMat result = *new cv::cuda::GpuMat();
				result.upload(temp);
				return result;
			}

			//from Tensor to GpuMat
			operator cv::cuda::GpuMat() {
				return make_gpumat();
			}
			#endif

		#pragma endregion

	#pragma endregion

	};


	#ifdef ON_USE_RANDOM
	template<typename Number>
	Tensor<Number>& randu(uint maj_span, uint min_span);


	Tensor<uchar> rand_ones_and_zeroes(uint maj_span, uint min_span);
	#endif

}