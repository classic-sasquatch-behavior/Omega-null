#pragma once
#include"language_manifold.h"
#include"omega_null.h"

namespace on {
	On_Structure Meta{

		On_Structure Parameter{
			static std::string output_path = "C:/Users/Thelonious/source/repos/Omega null/Omega null/on_language/src/interface/output";
			static std::string topdir = "C:/Users/Thelonious/source/repos/Omega null/";
		};

		On_Structure Configuration{
			const bool ON_EMBEDDED = false;
			const bool ON_FILE = true;
		};

	}
}
