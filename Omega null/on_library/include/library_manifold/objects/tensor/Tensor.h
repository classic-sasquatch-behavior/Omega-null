#pragma once
#include"global_manifold.h"




namespace on {

	template <typename Number>
	struct Device_Ptr {
		
		Device_Ptr(uint spans_in[4], Number* device_data_in) {
			
			for(int i = 0; i < 4; i++) {
				spans[i] = spans_in[i];
				if (spans[i] > 1) num_dims++;
			}
			device_data = device_data_in;
		}

		uint num_dims = 0;
		uint spans[4] = {1, 1, 1, 1};

		__device__ uint maj() const {return spans[0]; }
		__device__ uint min() const {return spans[1]; }
		__device__ uint cub() const {return spans[2]; }
		__device__ uint hyp() const {return spans[3]; }

		Number* device_data = nullptr;

		__device__ Number& operator ()(int maj_pos, int min_pos = 0, int cub_pos = 0, int hyp_pos = 0) { return device_data[(((((maj_pos * min()) + min_pos) * cub()) + cub_pos) * hyp()) + hyp_pos]; }

	};





	template<typename Number>
	struct Tensor {

		operator Device_Ptr<Number>() { 
			switch(synced) {
				case true: desync(device); break;
				case false: switch(current){
					case host: sync();
					default: break;
				} break;
			}
			return(Device_Ptr<Number>(spans, device_data)); 
		}



	#pragma region data	

		Number* device_data = nullptr;
		Number* host_data = nullptr;

		//for debug purposes
		std::string name = "uninitialized";

		uint num_dims = 0;

		uint spans[4] = {1, 1, 1, 1};

		__host__ uint maj() const { return spans[0]; }
		__host__ uint min() const { return spans[1]; }
		__host__ uint cub() const { return spans[2]; }
		__host__ uint hyp() const { return spans[3]; }

		bool synced = false;
		on::host_or_device current = on::host;

	#pragma endregion

	#pragma region get data

		//get data
		__host__ Number& operator ()(int maj, int min = 0, int cub = 0, int hyp = 0) {
			switch(synced){
				case true: desync(host); break;
				case false: switch(current){
					case device: sync(); break;
					default: break;
				} break;
			}

			return host_data[(((((maj * min) + min) * cub) + cub) * hyp) + hyp];
		}

		int num_elements() { return maj() * min() * cub() * hyp(); }
		int bytesize() { return (num_elements() * sizeof(Number)); }

#pragma endregion

	#pragma region structors



		Tensor(Number constant = 0) {
			initialize_memory();
			fill_memory(constant);
			ready();
		}

		Tensor(std::vector<uint> in_spans, Number constant = 0, std::string in_name = "default") {
			name = in_name;
			int num_dims_in = in_spans.size();
			num_dims = 0;
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

		Tensor(const Tensor& input) {
			name = input.name + "_copy";
			On_Copy(spans, input.spans, 4);
			//spans = input.spans;
			num_dims = input.num_dims;
			host_data = input.host_data;
			device_data = input.device_data;
			ready();
		}

		void operator=(const Tensor& input){
			name = input.name + "_copy";
			On_Copy(spans, input.spans, 4);
			//spans = input.spans;
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
				spans[0] = input.size();

				//careful with this one, something about how it handles memory could be throwing things off with the destructor
				std::copy(input.begin(), input.end(), host_data);

				sync();
			}

			//to vector
			operator std::vector<Number>() { return std::vector<Number>(host_data, host_data + num_elements()); }

			//to pointer
			__host__ operator Number* () {
				return &((*this)(0));
			}

			//from numerical type

			__host__ operator Number () {
				return (*this)(0);
			}

			//to numerical type - maybe this could take the sum of the array? would therefore be consistent with the use case I want it for (casting a 0 dim Tensor to a number)

			//comparison to numerical type
			bool operator ==(Number to_compare){
				Number self = (*this)(0);
				return to_compare == self;
			}


		#pragma endregion

		#pragma region ArrayFire

			//from array
			void operator=(af::array& input) {
				
				num_dims = input.numdims();
				for (int i = 0; i < num_dims; i++) {
					spans[i] = input.dims(i);
				}

				desync(device);
				initialize_device_memory();
				cudaMemcpy((void*)device_data, (void*)input.device<Number>(), bytesize(), cudaMemcpyDeviceToDevice);
				input.unlock(); //probably a sloppy way to do this, but oh well
				sync();
			}

			//to array
			operator af::array() { return af::array((dim_t)maj(), (dim_t)min(), host_data); }

		#pragma endregion

		#pragma region OpenCV

			//from Mat to Tensor
			void operator=(cv::Mat input) {
				num_dims = input.dims;
				bool has_channels = (input.channels() > 1);
				num_dims += has_channels;

				spans[0] = input.rows;
				spans[1] = input.cols;
				spans[2] = input.channels();

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