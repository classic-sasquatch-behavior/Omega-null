#pragma once

#include"global_manifold.h"








namespace on {

	On_Structure Random {
		
		On_Process Initialize {
			
			static void curand_xor(int size, int seed, curandState* states);
		};
	}
}