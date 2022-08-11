#include"manifold.h"








//#ifdef IRRELEVANT
//
//
//template <typename Number> struct DevPtr
//{
//	typedef Number Datatype;
//	typedef int index_type;
//
//	enum { element_size = sizeof(Datatype) };
//
//	Number* internal_data;
//
//	__host__ __device__ DevPtr() : internal_data(0) {} //best guess: if the struct is initialized with no argument, use the constructor for the datatype which was specified? or no, it makes an array?
//	__host__ __device__ DevPtr(Number* input_data) : internal_data(input_data) {} //
//
//	__host__ __device__ size_t elementSize() const { return element_size; }
//	__host__ __device__ operator Number* () { return internal_data; } //overload the () operator to return the pointer to internal data
//	__host__ __device__ operator const Number* () const { return internal_data; }
//};
//
//template <typename Number> struct PtrStep : public DevPtr<Number>
//{
//
//	size_t step;
//
//	__host__ __device__ PtrStep() : step(0) {}
//	__host__ __device__ PtrStep(Number* input_data, size_t step_) : DevPtr<Number>(input_array), step(step_) {}
//
//
//	//returns a whole major span, which () below obtains the exact element from. seems confusing.
//	__host__ __device__ Number* get_element(int maj = 0) { return (Number*)((char*)(((DevPtr<Number>*)this)->data) + maj * step); }
//	__host__ __device__ const Number* get_element(int maj = 0) const { return (const Number*)((const char*)(((DevPtr<Number>*)this)->data) + maj * step); }
//
//
//
//	//fetch data
//	__host__ __device__ Number& operator ()(int maj, int min) { return get_element(maj)[min]; }
//	__host__ __device__ const Number& operator ()(int maj, int min) const { return get_element(maj)[min]; }
//};
//
////this is the CudaPtr class, which gets passed to the kernel.
////need to unwind this business with PtrStep. 
//template <typename Number> struct CudaPtr : public PtrStep<Number>
//{
//	int maj_span;
//	int min_span;
//
//	__host__ __device__ CudaPtr() : maj_span(0), min_span(0) {}
//	__host__ __device__ CudaPtr(int input_maj_span, int input_min_span, Number* input_data, size_t input_step)
//		: PtrStep<Number>(input_data, input_step), maj_span(input_maj_span), min_span(input_min_span) {}
//
//	template <typename AlsoNumber>
//	explicit CudaPtr(const CudaPtr<AlsoNumber>& d) : PtrStep<Number>((Number*)d.data, d.step), cols(d.cols), rows(d.rows) {}
//
//
//
//};
//
//namespace af {
//	class array {
//		//addition to the af::array class
//		//templated function, which overloads the typecast to the class CudaPtr. I'm not sure why const is there, nor where the actual function is defined.
//		template <typename Number> operator CudaPtr<Number>() const;
//
//	};
//}
//
//
//#endif
//
//#ifdef SUBSTRATE_USE_PTR
//template <typename Number> 
//struct CudaPtr 
//{
//	typedef Number Datatype;
//	typedef int index_type; //I want to remove this
//
//	enum { element_size = sizeof(Datatype) };
//
//	Number* internal_data;
//	size_t step;
//
//	int maj_span;
//	int min_span;
//
//	__host__ __device__ CudaPtr(int input_maj_span, int input_min_span, Number* input_data, size_t input_step) 
//		: internal_data(input_data), step(input_step), maj_span(input_maj_span), min_span(input_min_span) {}
//
//	__host__ __device__ CudaPtr() : internal_data(0), step(0), maj_span(0), min_span(0) {} //I want to remove this
//
//	__host__ __device__ size_t elementSize() const { return element_size; }
//	__host__ __device__ operator Number* () { return internal_data; } //overload the () operator to return the pointer to internal data
//	__host__ __device__ operator const Number* () const { return internal_data; }
//
//	//returns a whole major span, which () below obtains the exact element from. seems confusing.
//	__host__ __device__ Number* get_element(int maj = 0) { return (Number*)(((char*)(this->internal_data)) + maj * step); }
//	__host__ __device__ const Number* get_element(int maj = 0) const { return (const Number*)((const char*)(((DevPtr<Number>*)this)->data) + maj * step); }
//
//	//fetch data
//	__host__ __device__ Number& operator ()(int maj, int min) { return get_element(maj)[min]; }
//	__host__ __device__ const Number& operator ()(int maj, int min) const { return get_element(maj)[min]; }
//
//	template <typename DifferentNumber>
//	explicit CudaPtr(const CudaPtr<DifferentNumber>& d) 
//		: internal_data(d.internal_data), step(d.step), maj_span(d.maj_span), min_span(d.min_span) {} //is this what's doing the gpumat typecast? or is it supposed to be a type conversion?
//};
//
//
////doesnt work
//class Array : public af::array {
//	template <typename Number> operator CudaPtr<Number>() const;
//};
//#endif
//
//#ifndef SUBSTRATE_USE_PTR
//#ifndef SUBSTRATE_USE_TENSOR
//
//template<typename Type>
//__global__ kernel conway_life(Type* present, Type* future, int width, int height) {
//	GET_DIMS(col, row);
//	CHECK_BOUNDS(width, height);
//
//	Type alive = present[(col*height) + row];
//	Type living_neighbors = 0;
//	FOR_NEIGHBOR(n_col, n_row, width, height, col, row,
//		living_neighbors += present[(n_col * height) + n_row];
//	);
//
//	Type result = (living_neighbors >= (3 - alive)) && (living_neighbors <= 3);
//	future[(col* height) + row] = result;
//	return;
//}
//
//#endif
//#endif
//
//#ifdef SUBSTRATE_USE_PTR
//__global__ void conway_life_kernel(CudaPtr<uchar> present, CudaPtr<uchar> future) {
//	GET_DIMS(col, row);
//	CHECK_BOUNDS(present.maj_span, present.min_span);
//
//	uchar alive = present(col, row);
//	uchar living_neighbors = 0;
//	FOR_NEIGHBOR(n_col, n_row, present.maj_span, present.min_span, col, row,
//		living_neighbors += present(n_col, n_row);
//	);
//
//	uchar result = (living_neighbors >= (3 - alive)) && (living_neighbors <= 3);
//	future(col, row) = result; 
//	return;
//}
//
//#endif



__global__ kernel conway_life(on::Tensor<uchar> present, on::Tensor<uchar> future) {
	GET_DIMS(col, row);
	CHECK_BOUNDS(present.maj_span, present.min_span);

	uchar alive = present(col, row);
	uchar living_neighbors = 0;
	FOR_NEIGHBOR(n_col, n_row, present.maj_span, present.min_span, col, row,
		living_neighbors += present(n_col, n_row);
	);

	uchar result = (living_neighbors >= (3 - alive)) && (living_neighbors <= 3);
	future(col, row) = result;
	return;
}



uchar* d_present;
uchar* d_future;

int main() {
	srand(time(NULL));
	on::Tensor<uchar> present = on::rand_ones_and_zeroes(WIDTH, HEIGHT);
	on::Tensor<uchar> future(WIDTH, HEIGHT, (uchar)0);
	array buffer = af::constant(0, WIDTH, HEIGHT);
	array display(WIDTH, HEIGHT, 3);
	af::Window window(WIDTH, HEIGHT);

//#ifdef SUBSTRATE_USE_AF
//	array neighbors = af::constant(0, WIDTH, HEIGHT, u8);
//	array kernel = af::constant(1, 3, 3, u8);
//#endif
	
	bool running = true;
	int start_time = now_ms();

	while (running) {

		int current_time = now_ms();
		int wait_time = (1000 / FPS) - (current_time - start_time);

//#ifdef SUBSTRATE_USE_AF
//		neighbors = af::convolve(present, kernel);
//		future = ((neighbors - present) >= (3 - present)) && ((neighbors - present) <= 3);
//
//		af::sync();
//		present = future;
//
//		//convert result to displayable frame
//
//#endif
//
//#ifdef SUBSTRATE_USE_CUDA
//			
//	#ifdef SUBSTRATE_USE_PTR
//		d_present = present.device<uchar>();
//		d_future = future.device<uchar>();
//		af::sync();
//		conway_life_kernel<uchar><<<num_blocks, threads_per_block>>> (d_present, d_future, WIDTH, HEIGHT);
//	#endif
//
//	#ifdef SUBSTRATE_USE_PTR
//		conway_life_kernel <<<num_blocks, threads_per_block>>> (present, future, WIDTH, HEIGHT);
//	#endif
//
//#ifndef SUBSTRATE_USE_TENSOR
//		SYNC_AND_CHECK_FOR_ERRORS(conway_life_kernel);
//		present.unlock();
//		future.unlock();
//		present = future;
//#endif
		on::Kernel::conf_2d(WIDTH, HEIGHT);
		conway_life<<<KERNEL_SHAPE>>>(present, future);
		SYNC_AND_CHECK_FOR_ERRORS(conway_life);
		present = future;


		//buffer.write(const_cast<const uchar*>(present.device_data), present.bytesize(), afDevice);
		//buffer = af::constant(0, WIDTH, HEIGHT);
		buffer = (af::array)present;



		gfor(seq i, 3) {
			display(span, span, i) = buffer * 255;
		}

		window.image(display);

		std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));
		start_time = now_ms();
		std::cout << "FPS: " << 1000/wait_time << std::endl;
	}



	return 0;
}