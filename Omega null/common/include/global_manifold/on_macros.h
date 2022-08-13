#pragma once


//#define __kernel__(_xml_object_name_, _content_) XMLObject _xml_object_name_ << (#_content_)

inline int now_ms() { return std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count(); }

#define IGNORE_DIM 1

#define ON_STRING(_content_) #_content_

//a struct which contains only data
#define On_Being struct

//a static struct which contains only functions
#define On_Process static struct

//a namespace which contains both data and functions
#define On_Structure namespace

//#define __begin_kernel__
//
//#define __end_kernel__



