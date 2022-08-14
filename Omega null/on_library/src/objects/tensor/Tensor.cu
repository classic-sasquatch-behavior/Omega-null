#include"global_manifold.h"
#include"on_library.h"

#ifdef ON_USE_RANDOM
namespace on {


	__global__ kernel seed_curand(int num_elements, int seed, curandState_t* states) {
		GET_DIMS(id, zero);
		CHECK_BOUNDS(num_elements, IGNORE_DIM);
		curand_init(seed, id, 0, &states[id]);
	}


	template<typename Number>
	__global__ kernel generate_random_uniform(curandState_t* states, on::Tensor<Number> output) {
		GET_DIMS(col, row);
		CHECK_BOUNDS(output.maj_span, output.min_span);
		float result = curand_uniform(&states[(col * output.min_span) + row]);
		output(col, row) = (Number)result;
	}

	__global__ kernel generate_random_ones_and_zeroes(curandState_t* states, on::Tensor<uchar> output) {
		GET_DIMS(col, row);
		CHECK_BOUNDS(output.maj_span, output.min_span);
		float rand = curand_uniform(&states[(col * output.min_span) + row]);
		uchar result = (rand > 0.5);
		output(col, row) = result;
	}




	//template<typename Number>
	//Tensor<Number>& randu(uint maj_span, uint min_span) {

	//	uint num_elements = maj_span * min_span;
	//	curandGenerator_t* generator;
	//	curandState_t* states;
	//	on::Tensor<Number> result(maj_span, min_span, (Number)0);
	//	int seed = rand();

	//	conf_1d(num_elements);
	//	seed_curand <<<KERNEL_SHAPE>>> (num_elements, seed, states);
	//	SYNC_AND_CHECK_FOR_ERRORS(seed_curand);

	//	conf_2d(maj_span, min_span);
	//	generate_random_uniform<Number> <<<KERNEL_SHAPE>>> (states, result);
	//	SYNC_AND_CHECK_FOR_ERRORS(generate_random_uniform);

	//	curandDestroyGenerator(*generator);
	//	return result;
	//}


	Tensor<uchar> rand_ones_and_zeroes(uint maj_span, uint min_span) {

		uint num_elements = maj_span * min_span;
		//curandGenerator_t* generator;
		curandState_t* states;
		cudaMalloc(&states, maj_span * min_span * sizeof(curandState_t));
		on::Tensor<uchar> result(maj_span, min_span, (uchar)0);

		int seed = rand();

		on::Kernel::conf_1d(num_elements);
		seed_curand <<<KERNEL_SHAPE>>> (num_elements, seed, states);
		SYNC_AND_CHECK_FOR_ERRORS(seed_curand);

		on::Kernel::conf_2d(maj_span, min_span);
		generate_random_ones_and_zeroes<<<KERNEL_SHAPE>>> (states, result);
		SYNC_AND_CHECK_FOR_ERRORS(generate_random_ones_and_zeroes);


		cudaFree(states);
		//curandDestroyGenerator(*generator);
		return result;
	}


}
#endif