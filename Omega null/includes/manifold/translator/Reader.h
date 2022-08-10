#pragma once
#include"header_manifold.h"

ON_STRUCTURE Reader{
	using ON_STRUCTURE Meta;
	using ON_STRUCTURE Writer;

	ON_PROCESS Read{
		static void next_node(Node root, KernelData data, std::ofstream & cuda_file) {

		std::string structure_type = root.name();

		cuda_file << Get::tabs(data.layer);
		if (structure_type == "Kernel") { Write::kernel_declaration(root, data, cuda_file); }
		if (structure_type == "For_Element") { Write::for_element(root, data, cuda_file); }
		if (structure_type == "For_Neighbor") { Write::for_neighbor(root, data, cuda_file); }
		if (structure_type == "For_Maj") { Write::for_maj(root, data, cuda_file); }
		if (structure_type == "For_Min") { Write::for_min(root, data, cuda_file); }
		if (structure_type == "Cast_Down") { Write::cast_down(root, data, cuda_file); }
		if (structure_type == "Cast_Up") { Write::cast_up(root, data, cuda_file); }

		for (Node node : root) {
			if (node.type() == pugi::node_pcdata) { Write::text_to_kernel(cuda_file, node.text().as_string()); }
			else { Read::next_node(node, data, cuda_file); }
		}

		data.layer--;
		Write::to_kernel(data, cuda_file, "}\n");
		}
	};
};