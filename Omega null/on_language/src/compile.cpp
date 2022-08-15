#include"global_manifold.h"
#include"on_language.h"

/*
	had to include this to the .vcxproj:

  <ItemGroup>
	<ClInclude Include="output\*.h" />
	<CudaCompile Include="output\*.cu" />
  </ItemGroup>

  but, it doesn't work as automatically as I'd like it to. we'll do it manually for now and 
  figure out how to make this work later.
*/

using namespace on;
using On_Structure on::Meta;
using On_Structure on::Writer;
using On_Structure on::Loader;
using On_Structure on::Reader;
void compile() {

	Load::project(Parameter::topdir, LoadStructures::file_queue);
	Load::structures(LoadStructures::file_queue);


	

	//4) find and replace #include"name.on" with #include"name.h"


}