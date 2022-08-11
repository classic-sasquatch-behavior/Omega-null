#pragma once

#include<iostream>
#include <chrono>
#include <thread>

#include<arrayfire.h>
using af::array;
using af::seq;
using af::span;

#include<useful_cuda_macros.h>

#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<curand.h>
#include<curand_kernel.h>

#undef min