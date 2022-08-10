#include"manifold.h"

/*
	had to include this to the .vcxproj:

  <ItemGroup>
	<ClInclude Include="output\*.h" />
	<CudaCompile Include="output\*.cu" />
  </ItemGroup>

  but, it doesn't work as automatically as I'd like it to. we'll do it manually for now and 
  figure out how to make this work later.
*/

using ON_STRUCTURE Meta;
using ON_STRUCTURE Writer;
using ON_STRUCTURE Loader;
using ON_STRUCTURE Reader;
int main() {

	Load::project(Parameter::topdir, LoadStructures::file_queue);
	Load::structures(LoadStructures::file_queue);

	//4) find and replace #include"name.on" with #include"name.h"

	return 0;
}