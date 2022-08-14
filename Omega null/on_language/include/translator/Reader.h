#pragma once
#include"global_manifold.h"
#include"on_language.h"

namespace on {
	On_Structure Reader{
		On_Process Read {
			static void next_node(Node root, KernelData data, std::ofstream & cuda_file);
		};
	};
}
