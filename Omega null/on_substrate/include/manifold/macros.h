#pragma once


#define __kernel__(_xml_object_name_, _content_) XMLObject _xml_object_name_ << (#_content_)

inline int now_ms() { return std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count(); }


#define IGNORE_DIM 1






