#include"../../global_includes.h"




namespace on {


	struct Structure;

	Parser::Parser() {
		initialize_page();
	}

	void Parser::generate_report() {
		std::cout << std::endl;
		std::cout << "generated report:" << std::endl;
		for (std::string line : _content) {
			std::cout << line << std::endl;
		}
		std::cout << std::endl;
	}

#pragma region reader functions
	void Parser::initialize_page() {
		Structure* page = nullptr;
		page = new Structure(page, 0);
		page->set_name("page");
		_structures.push_back(page);
		set_current_structure(page);
		_buffer = "page";
		flush_buffer(1);
	}

	void Parser::initialize_structure() {
		Structure* new_structure = new Structure(get_current_structure(), level + 1);
		add_structure(new_structure);
		set_current_structure(new_structure);
	}

	std::string Parser::flush_buffer(int argument) {
		std::string begin = "@begin ";
		std::string end = "@end ";

		switch (argument) {
		case RETURN: break;
		case END_RAW_TEXT: _content.push_back(_buffer); _current_structure->add_content(_buffer);  break;
		case MARK_BEGINNING_OF_STRUCTURE: begin += _buffer; _content.push_back(begin); break;
		case MARK_END_OF_STRUCTURE: end += _buffer; _content.push_back(end); break;
		case PUSH_BACK_DIM:_current_structure->add_dim_name(_buffer); break;
		default: std::cout << "ERROR - flush_buffer argument undefined: " << argument << std::endl; break;
		}
		std::string result = _buffer;
		_buffer.clear();
		return result;
	}

	void Parser::change_state(int new_state) {
		int old_state = state;
		switch (new_state) {
		case IDLE: break;
		case RAW_TEXT: break;
		case OPEN_TAG: break;
		case STRUCTURE_TYPE: break;
		case STRUCTURE_NAME: break;
		case STRUCTURE_DIMS: break;
		case STRUCTURE_DATA: break;
		case CLOSE_TAG: break;
		case TAG_IDLE: break;
		default: std::cout << "ERROR - undefined new state: " << new_state << std::endl; break;
		}

		switch (old_state) {
		case IDLE: break;
		case RAW_TEXT: break;
		case OPEN_TAG: break;
		case STRUCTURE_TYPE: break;
		case STRUCTURE_NAME: break;
		case STRUCTURE_DIMS: break;
		case STRUCTURE_DATA: break;
		case CLOSE_TAG: break;
		case TAG_IDLE: break;
		default: std::cout << "ERROR - undefined old state: " << old_state << std::endl; break;
		}
		state = new_state;
	}

	void Parser::complete_type() {
		_current_structure->set_type(flush_buffer(RETURN)); 
	}

	void Parser::complete_name() {
		_current_structure->set_name(flush_buffer(MARK_BEGINNING_OF_STRUCTURE));
	}

	void Parser::complete_dims() {
		_current_structure->set_num_dims(_current_structure->get_all_dims().size());
	}

	void Parser::complete_data() {
		std::string data = flush_buffer(RETURN);
		_current_structure->set_data(data);
		complete_parent(data);
	}

	void Parser::complete_parent(std::string data_to_check) {
		std::string parent;

		//split on spaces
		int start = 0;
		int end = data_to_check.find(",");
		parent = data_to_check.substr(start, end);
		
		start = parent.find_first_not_of(" ");
		end = parent.find_last_not_of(" ");
		parent = parent.substr(start, end);

		std::string parent_name;
		std::string parent_type;

		start = 0;
		end = parent.size();
		int split = parent.find(" "); //this is not at all robust. specifically, will break if the arguments have any additional spaces. you probably want to refactor the whole function eventually, remember when u make the semantic checker.
		parent_type = parent.substr(start, split);
		parent_name = parent.substr(split, end);


	}

	void Parser::close_tag() {

	}

	void Parser::wrap_up() {
		//stuff goes here
	}
#pragma endregion

#pragma region states
	void Parser::read(char input) {
		std::cout << input << " : " << state << " | " << _buffer << std::endl;
		switch (state) {
		case IDLE:
			idle_state(input); break;
		case RAW_TEXT:
			raw_text_state(input); break;
		case OPEN_TAG:
			open_tag_state(input); break;
		case STRUCTURE_TYPE:
			structure_type_state(input); break;
		case STRUCTURE_NAME:
			structure_name_state(input); break;
		case STRUCTURE_DIMS:
			structure_dims_state(input);  break;
		case STRUCTURE_DATA:
			structure_data_state(input);  break;
		case CLOSE_TAG:
			close_tag_state(input);  break;
		case TAG_IDLE:
			tag_idle_state(input); break;
		default:
			__THROW_ERROR(read, undefined state, state); break;
		}
	}

	//state 0
	void Parser::idle_state(char input) {
		switch (input) {
		case '<': change_state(OPEN_TAG); break;
		case ' ': break;
		case '\n': break;
		case '\t': break;
		default: change_state(RAW_TEXT); _buffer += input; break;
		}
	}

	//state 1
	void Parser::raw_text_state(char input) {
		switch (input) {
		case '\n': flush_buffer(END_RAW_TEXT); change_state(IDLE); break;
		default: _buffer += input;  break;
		}
	}

	//state 2
	void Parser::open_tag_state(char input) {
		switch (input) {
		case '/': change_state(CLOSE_TAG); break;
		default: change_state(STRUCTURE_TYPE); add_to_buffer(input); initialize_structure(); break;
		}
	}

	//state 8
	void Parser::tag_idle_state(char input) {
		switch (input) {
		case '{': change_state(STRUCTURE_DATA); break;
		case '(': change_state(STRUCTURE_DIMS); break;
		case '>': close_tag();  change_state(IDLE); break;
		case ' ': break;
		case ',': break;
		default: add_to_buffer(input); change_state(STRUCTURE_NAME);  break;
		}
	}

	//state 3
	void Parser::structure_type_state(char input) { 
		switch (input) {
		case ' ': if (!_buffer.empty()) { complete_type(); change_state(TAG_IDLE);} break;
		case ',': complete_type(); change_state(TAG_IDLE); break;
		case '(': complete_type(); change_state(STRUCTURE_DIMS); break;
		case '>': complete_type(); change_state(IDLE); break;
		default: _buffer += input; break;
		}
	}

	//state 4
	void Parser::structure_name_state(char input) {
		switch (input) {
		case ' ': break;
		case ',': complete_name(); change_state(TAG_IDLE); break;
		case '(': complete_name(); change_state(STRUCTURE_DIMS); break;
		case '>': complete_name(); change_state(IDLE); break;
		default: _buffer += input; break;
		}
	}

	//state 5
	void Parser::structure_dims_state(char input) {
		switch (input) {
		case ',': flush_buffer(PUSH_BACK_DIM);  break;
		case ')': flush_buffer(PUSH_BACK_DIM); complete_dims(); change_state(TAG_IDLE); break;
		case ' ': break;
		default: add_to_buffer(input); break;
		}
	}

	//state 6
	void Parser::structure_data_state(char input) {
		switch (input) {
		case '}': complete_data(); change_state(TAG_IDLE); break;
		default: add_to_buffer(input); break;
		}
	}

	//state 7
	void Parser::close_tag_state(char input) { 
		switch (input) {
		case ' ': break;
		case '>': flush_buffer(MARK_END_OF_STRUCTURE); step_down(); set_current_structure(_current_structure->get_parent()); change_state(IDLE); break;
		default: _buffer += input; break;
		}
	}
#pragma endregion

#pragma region writer functions

	void Parser::generate_code() {
		std::vector<Structure*> kernels;
		tokenize_content(kernels);
		if (kernels.size() <= 0) { __THROW_ERROR(generate code, no kernels found, "very sad"); }

		std::ofstream header_file;
		std::ofstream cuda_file;
		std::string kernel_name = kernels[0]->get_name();
		initialize_files(kernel_name, header_file, cuda_file);

		for (Structure* kernel : kernels) {
			generate_header(kernel, header_file);
			generate_cuda_code(kernel, cuda_file);
		}

		header_file.close();
		cuda_file.close();



	}

	void Parser::generate_header(Structure* kernel, std::ofstream& header_file) {
		std::string data = kernel->get_data();
		std::string name = kernel->get_name();

		header_file << "void " + name + "_launch(" + data + ");";
	}

	void Parser::generate_cuda_code(Structure* kernel, std::ofstream& cuda_file) {
		cuda_file << template_kernel_begin(kernel);
		cuda_file << template_kernel_content(kernel); //TODO: fill this out 
		cuda_file << template_kernel_end(kernel);

		cuda_file << template_launch_begin(kernel);
		cuda_file << template_launch_dims(kernel);
		cuda_file << template_launch_kernel_call(kernel);
		cuda_file << template_launch_end(kernel);
	}

#pragma region code templates
	std::string Parser::template_kernel_begin(Structure* kernel) {
		std::string data = kernel->get_data();
		std::string name = kernel->get_name();

		std::string result = "__global__ void " + name + "_kernel(" + data + "){\n";
		return result;
	}

	std::string Parser::template_kernel_content(Structure* kernel) {
		std::string result = "";
		return result;
	}

	std::string Parser::template_kernel_end(Structure* kernel) {
		std::string result = "}\n\n";
		return result;
	}

	std::string Parser::template_launch_begin(Structure* kernel) {
		std::string data = kernel->get_data();
		std::string name = kernel->get_name();

		std::string result = "void " + name + "_launch(" + data + "){\n";
		return result;
	}

	std::string Parser::template_launch_dims(Structure* kernel) {
		//need shape. shape comes from the first argument within data. probably want to get it when we search for kernels in the last step.
		std::string parent = kernel->get_parent_name();
		std::string result = "kernel dims\n";
		

		return result;
	}

	std::string Parser::template_launch_kernel_call(Structure* kernel) {
		std::string kernel_name = kernel->get_name();
		std::string result = "kernel call\n";

		return result;
	}

	std::string Parser::template_launch_end(Structure* kernel) {
		std::string result = "}";
		return result;
	}

#pragma endregion










	void Parser::tokenize_content(std::vector<Structure*> &kernels) {

		std::string space = " ";

		for (std::string entry : _content) {
			std::vector<std::string> split_entry;

			//split on spaces
			int start = 0;
			int end = entry.find(space);
			while (end != -1) {
				std::string word = entry.substr(start, end - start);
				split_entry.push_back(word);
				start = end + space.size();
				end = entry.find(space, start);
			}
			std::string word = entry.substr(start, end - start);
			split_entry.push_back(word);



			if (split_entry[0] == "@begin") {
				std::string structure_name = split_entry[1];
				Structure* this_structure = find_structure_with_name(structure_name);
				if (this_structure->get_type() == "Kernel") {
					kernels.push_back(this_structure);
				}
				
			}
			else { continue; }
		}

	}

	void Parser::identify_kernels() {
		
	}

	void Parser::initialize_files(std::string kernel_name, std::ofstream &header_file, std::ofstream &cuda_file) {
		std::string header_name = kernel_name + ".h";
		std::string cuda_file_name = kernel_name + ".cu";

		header_file.open(output_path + header_name);
		cuda_file.open(output_path + cuda_file_name);

	}

	Structure* Parser::find_structure_with_name(std::string target_name) {
		Structure* result = nullptr;
		for (Structure* structure : _structures) {
			std::string current_name = structure->get_name();
			if (current_name == target_name) {
				result = structure;
				break;
			}
		}
		if (result == nullptr) {
			__THROW_ERROR(find_structure_with_name, no structure with name, target_name);
		}
		return result;
	}

#pragma endregion


}