
__global__ void sample_centers(Tensor source, Tensor center_rows, Tensor center_cols){
	GET_DIMS(sector_row, sector_col);
	CHECK_BOUNDS(source.maj_span, source.min_span);
	
      center_rows(sector_row, sector_col) = actual_row;
      center_cols(sector_row, sector_col) = actual_col;
    }

}

void sample_centers_launch(Tensor source, Tensor center_rows, Tensor center_cols){
	on::Tensor& shape = source;
	
	unsigned int block_dim_x = -1;
	unsigned int block_dim_y = -1;
	unsigned int grid_dim_x = (shape.maj_span - (shape.maj_span % block_dim_x))/block_dim_x;
	unsigned int grid_dim_y = (shape.min_span - (shape.min_span % block_dim_y))/block_dim_y;
	
	dim3 num_blocks(grid_dim_x + 1, grid_dim_y + 1);
	dim3 threads_per_block(block_dim_x, block_dim_y)
	
	sample_centers<<<num_block, threads_per_block>>>(Tensor source, Tensor center_rows, Tensor center_cols);
}
__global__ void assign_pixels_to_centers(Tensor L_src, Tensor A_src, Tensor B_src, int intensity_modifier, Tensor center_rows, Tensor center_cols, Tensor labels){
	GET_DIMS(row, col);
	CHECK_BOUNDS(labels.maj_span, labels.min_span);

    int self_channels[3] = { L_src(row, col), A_src(row, col), B_src(row, col) };
    int self_coordinates[2] = {row, col};
    
    	
      
      int min_distance = 1000000;
      int closest_center_id = -1;
      
      	
        int actual_neighbor_row = center_rows(neighbor_row, neighbor_col);
        int actual_neighbor_col = center_cols(neighbor_row, neighbor_col);

        int neighbor_channels[3] = { L_src(actual_neighbor_row, actual_neighbor_col), A_src(actual_neighbor_row, actual_neighbor_col), B_src(actual_neighbor_row, actual_neighbor_col) };
        int neighbor_coordinates[2] = {actual_neighbor_row, actual_neighbor_col};
        int neighbor_label = LINEAR_CAST(neighbor_row, neighbor_col, center_rows.min_span);
        
        int color_distance = 0;
        UNROLL_FOR(int i = 0; i }

}

}

void assign_pixels_to_centers_launch(Tensor L_src, Tensor A_src, Tensor B_src, int intensity_modifier, Tensor center_rows, Tensor center_cols, Tensor labels){
	on::Tensor& shape = labels;
	
	unsigned int block_dim_x = -1;
	unsigned int block_dim_y = -1;
	unsigned int grid_dim_x = (shape.maj_span - (shape.maj_span % block_dim_x))/block_dim_x;
	unsigned int grid_dim_y = (shape.min_span - (shape.min_span % block_dim_y))/block_dim_y;
	
	dim3 num_blocks(grid_dim_x + 1, grid_dim_y + 1);
	dim3 threads_per_block(block_dim_x, block_dim_y)
	
	assign_pixels_to_centers<<<num_block, threads_per_block>>>(Tensor L_src, Tensor A_src, Tensor B_src, int intensity_modifier, Tensor center_rows, Tensor center_cols, Tensor labels);
}
