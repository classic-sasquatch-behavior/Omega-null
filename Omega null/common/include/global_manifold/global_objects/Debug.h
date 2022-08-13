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

	On_Being Error{
		Error() {

		};
		Debug::strangeness strangeness = unknown;
		std::string comment = "no comment.";
		std::string file = "unspecified";
		uint line = 0;
	};

		On_Process Print {

		};

		On_Process Throw {
			static void error(Debug::Error& error) {
				switch (error.strangeness) {
				case likely:
				case plausible:
				case unknown:
				case unlikely:
				case impossible:
				default: error(*new Error(strangeness::impossible, "...what did you say to me?", __FILE__, __LINE__)); break; //let's not start that again
				}




			}
		}














	}
}





