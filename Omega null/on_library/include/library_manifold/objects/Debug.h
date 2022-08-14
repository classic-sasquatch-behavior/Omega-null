#pragma once
#include"global_manifold.h"


namespace on {
	On_Structure Debug {
		enum strangeness {
			likely = 0,
			plausible = 1,
			unknown = 2,
			unlikely = 3,
			impossible = 4
		};

		On_Being Error {
			Debug::strangeness strangeness;
			std::string comment;
			std::string file;
			uint line;
			Error() : comment(""), strangeness(unknown), file("unspecified"), line(0) {}
		};

		On_Process Throw {
			static void error(Debug::Error & error) {
				std::string message = "ON::ERROR : ";
				message += error.comment;
				message += " : this happened in file : ";
				message += error.file;
				message += " : at line : ";
				message += error.line;

				switch (error.strangeness) {
					case likely: message += " : you knew that would happen [0/4]"; break;
					case plausible: message += " : you thought that might happen [1/4]"; break;
					case unknown: message += " : you weren't sure if that would happen [2/4]"; break;
					case unlikely: message += " : you didn't think that would happen [3/4]"; break;
					case impossible: message += " : that definitely wasn't supposed to happen [4/4]"; break;
				default: message += " : ...what? [?/ 4]"; break;
				}
				std::cout << "\n" + message + "\n";
			}
		};

	}
}





