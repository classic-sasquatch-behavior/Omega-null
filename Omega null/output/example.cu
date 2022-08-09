
__global__ void add_by_element(on::Tensor A, on::Tensor B, on::Tensor C){
GET_DIMS(row, col);
CHECK_BOUNDS(A.maj_span, A.min_span);

		C(row, col) = A(row, col) + B(row, col);


}

void add_by_element_launch(on::Tensor A, on::Tensor B, on::Tensor C)
on::Tensor& shape = A;

unsigned int block_dim_x = -1;
unsigned int block_dim_y = -1;
unsigned int grid_dim_x = (shape.maj_span - (shape.maj_span % block_dim_x))/block_dim_x;
unsigned int grid_dim_y = (shape.min_span - (shape.min_span % block_dim_y))/block_dim_y;
dim3 num_blocks(grid_dim_x + 1, grid_dim_y + 1);
dim3 threads_per_block(block_dim_x, block_dim_y)
add_by_element<<<num_block, threads_per_block>>>(on::Tensor A, on::Tensor B, on::Tensor C);
}
