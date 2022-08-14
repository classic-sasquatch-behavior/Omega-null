#pragma once





ON_STRUCTURE Reader{
	ON_PROCESS Read {
		static void next_node(Node root, KernelData data, std::ofstream & cuda_file);
	};
};