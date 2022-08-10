
void sample_centers_launch(Tensor source, Tensor center_rows, Tensor center_cols);
void assign_pixels_to_centers_launch(Tensor L_src, Tensor A_src, Tensor B_src, int intensity_modifier, Tensor center_rows, Tensor center_cols, Tensor labels);
