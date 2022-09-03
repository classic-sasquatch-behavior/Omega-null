#pragma once

#include"global_manifold.h"








namespace on {

	On_Structure Random {
		
		On_Process Initialize {
			
			static curandState* curand_xor(int size, int seed);
		};
	}
}