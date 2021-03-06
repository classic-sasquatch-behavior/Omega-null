#pragma once
#include"../../headers/global_includes.h"


namespace on {

	struct Structure {
	public:
		Structure(Structure* parent, int level);
		~Structure();

#pragma region get-set
		void set_level(int input) { _level = input; }
		void set_type(std::string input) { _type = input; }
		void set_name(std::string input) { _name = input; }
		void set_data(std::string input) { _data = input; }  
		void set_parent(Structure* input) { _parent = input; }
		void set_parent_name(std::string input) { _parent_name = input; }
		void set_parent_type(std::string input) { _parent_type = input; }
		void set_num_dims(int input) { _num_dims = input; }
		void add_dim_name(std::string input) { _dim_names.push_back(input); }
		void add_child(Structure* input) { _children.push_back(input); }
		void add_content(std::string input) { _content.push_back(input); }

		int get_level() { return _level; }
		int get_num_dims() { return _num_dims; }
		std::string get_type() { return _type; }
		std::string get_name() { return _name; }
		std::string get_data() { return _data; }
		Structure* get_parent() { return _parent; }
		std::string get_parent_name() { return _parent_name; }
		std::string get_parent_type() { return _parent_type; }
		
		std::string get_dim(int index) { return _dim_names[index]; }
		std::vector<std::string> get_all_dims() { return _dim_names; }
		Structure* get_child(int index) { return _children[index]; }
		std::string get_line_of_content(int index) { return _content[index]; }
#pragma endregion

	private:
		int _level = 0;
		std::string _type = "default";
		std::string _name = "default";
		std::string _data = " ";		//data i.e. "A, B, C" as a raw string
		Structure* _parent = nullptr;
		std::string _parent_name;
		std::string _parent_type;
		int _num_dims;
		std::vector<std::string> _dim_names;
		std::vector<Structure*> _children;
		std::vector<std::string> _content;

	};

}