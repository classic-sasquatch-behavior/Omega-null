#pragma once


namespace on {

	#define FOR_NXN(_conv_from_, _conv_to_, _base_shape_, _base_maj_, base_min_, _new_maj_, _new_min_, _content_)								 \
		(																																		 \
			for(int _i_maj_ = _conv_from_; _i_maj_ <= _conv_to_; _i_maj_++){																	 \
				for (int _i_min_ = -_conv_from_; _i_min_ <= _conv_to_; i_min++) {																 \
					int _new_maj_ = _i_maj_ + _base_maj_;																						 \
					int _new_min_ = _i_min_ + _base_min_;																						 \
					if((_neighbor_maj_ < 0)||(_neighbor_min_ < 0)||(_neighbor_maj_ >= _shape_.maj)||(_neighbor_min_ >= _shape_.min) { continue;} \
					(_content_);																												 \
				}																																 \
			}																																	 \
		)

	#define FOR_NXN_EXCLUSIVE(_conv_shape_, _base_shape_, _base_maj_, base_min_, _new_maj_, _new_min_, _content_)					\
		(																															\
			for(int _i_maj_ = -_conv_from_; _i_maj_ <= _conv_to_; _i_maj_++){														\
				for (int _i_min_ = -_conv_from_; _i_min_ <= _conv_to_; i_min++) {													\
					int _new_maj_ = _i_maj_ + _base_maj_;																			\
					int _new_min_ = _i_min_ + _base_min_;																			\
					if((_neighbor_maj_ < 0)||(_neighbor_min_ < 0)||(_neighbor_maj_ >= _shape_.maj)||(_neighbor_min_ >= _shape_.min) \
						||((_neighbor_maj_ == )&&( _neighbor_min_ == ))){ continue;}												\
					(_content_);																									\
				}																													\
			}																														\
		)

}


