#pragma once
#include"language_manifold.h"
#include"omega_null.h"

namespace on {
	On_Structure Reader{
		On_Process Read {
			static void next_node(Node root, KernelData data, std::ofstream & cuda_file);
		};
	};
}
