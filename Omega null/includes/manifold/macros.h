#pragma once


typedef unsigned int uint;



#define ON_STRING(_content_) #_content_

//a struct which contains only data
#define ON_BEING struct

//a static struct which contains only functions
#define ON_PROCESS static struct

//a namespace which contains both data and functions
#define ON_STRUCTURE namespace