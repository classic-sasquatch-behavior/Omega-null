#pragma once
#include"global_manifold.h"




namespace on {

	template<typename Number>
	struct Tensor {

	#pragma region data	

		Number* device_data = nullptr;
		Number* host_data = nullptr;

		uint num_dims = 0;
		std::vector<uint> spans = { 1, 1, 1, 1 };

		uint& maj_span = spans[0];
		uint& min_span = spans[1];
		uint& cub_span = spans[2];
		uint& hyp_span = spans[3];

		bool synced = false;
		on::host_or_device current = host;

	#pragma endregion

	#pragma region get data

		//get data
		__host__ __device__ Number& operator ()(int maj, int min = 0, int cub = 0, int hyp = 0) {
			#ifdef __CUDA_ARCH__			
				return device_data[(((((maj * min_span) + min) * cub_span) + cub) * hyp_span) + hyp];
			#else
				return host_data[(((((maj * min_span) + min) * cub_span) + cub) * hyp_span) + hyp];
			#endif
		}

		int num_elements() { return maj_span * min_span * cub_span * hyp_span; }
		int bytesize() { return (num_elements() * sizeof(Number)); }

#pragma endregion

	#pragma region structors

		Tensor(Number constant = 0) {
			initialize_memory();
			fill_memory(constant);
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

		~Tensor() {
			cudaFree(device_data);
			free(host_data); //should be free, because I used malloc in initialize memory... right? 
							 //evidently not, because I still get that heap error. that or theres another problem.
							 //the fact that I'm just cutting and running in the main function probbaly doesnt help.
		}

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

	#pragma endregion

	#pragma region memory functions

		#pragma region initialize memory
		void initialize_memory() {
			initialize_host_memory();
			initialize_device_memory();
		}

		void initialize_host_memory() {
			host_data = (Number*)malloc(bytesize());
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
		}

		#pragma endregion

		#pragma region transfer memory

		//bytesize will always be accurate as long as the meta data is accurate, but can cuda memcpy allocate space on its own, or should I be managing 
		//the memory completely manually? (i.e. freeing and reallocating it every time).
		//CUDA has gotten pretty good about stuff like that recently so for now I wont worry about it unless it becomes a problem. If you start getting
		//I guess device memory leaks, start here by controlling the memory more carefully.
		void upload() {
			cudaMemcpy(device_data, host_data, bytesize(), cudaMemcpyHostToDevice);
		}

		void download() {
			cudaMemcpy(host_data, device_data, bytesize(), cudaMemcpyDeviceToHost);
		}

		#pragma endregion

	#pragma endregion



	#pragma region interop

		#pragma region omega null

			//is this actually already implicit?
			void operator=(Tensor input) {
				maj_span = input.maj_span;
				min_span = input.min_span;
				host_data = input.host_data;
				device_data = input.device_data;
			}

		#pragma endregion

		#pragma region std

			//from vector
			void operator=(std::vector<Number> input) {
				desync(host);
				num_dims = 1;
				spans[0] = input.size();

				//careful with this one, something about how it handles memory could be throwing things off with the destructor
				std::copy(input.begin(), input.end(), host_data);

				sync();
			}

			//to vector
			operator std::vector<Number>() { return std::vector<Number>(host_data, host_data + num_elements()); }

			//from numerical type


			//to numerical type - maybe this could take the sum of the array? would therefore be consistent with the use case I want it for (casting a 0 dim Tensor to a number)

		#pragma endregion

		#pragma region ArrayFire

			//from array
			void operator=(af::array& input) {
				maj_span = input.dims(0);
				min_span = input.dims(1);
				desync(device);
				device_data = input.device<Number>(); //would cause problems with arrayfire backend, which is why we call unlock below
				sync();
				input.unlock(); //probably a sloppy way to do this, but oh well
			}

			//to array
			operator af::array() { return af::array((dim_t)maj_span, (dim_t)min_span, host_data); }

		#pragma endregion

		#pragma region OpenCV

			//from Mat to Tensor
			void operator=(cv::Mat input) {
				num_dims = input.dims;
				bool has_channels = (input.channels > 1);
				num_dims += has_channels;

				spans[0] = input.dims;
				spans[1] = input.cols;
				spans[2] = input.channels;

				desync(host);
				host_data = (Number*)input.data;
				sync();
			}

			//from Tensor to Mat
			operator cv::Mat() {
				return cv::Mat(spans[0], spans[1], cv::DataType<Number>::type, host_data);
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