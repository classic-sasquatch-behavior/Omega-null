#pragma once




inline int now_ms() { return std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count(); }


//a struct which contains only data
#define On_Being struct

//a static struct which contains only functions
#define On_Process static struct

//a namespace which contains both data and functions
#define On_Structure namespace







