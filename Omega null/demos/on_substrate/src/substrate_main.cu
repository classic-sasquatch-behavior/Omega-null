#include"global_manifold.h"
#include"omega_null.h"







#ifdef ILL_FIX_IT_LATER
__global__ kernel conway_life(on::Tensor<uchar> present, on::Tensor<uchar> future) {
	GET_DIMS(col, row);
	CHECK_BOUNDS(present.maj_span, present.min_span);

	uchar alive = present(col, row);
	uchar living_neighbors = 0;
	FOR_NEIGHBOR(n_col, n_row, present.maj_span, present.min_span, col, row,
		living_neighbors += present(n_col, n_row);
	);

	uchar result = (living_neighbors >= (3 - alive)) && (living_neighbors <= 3);
	future(col, row) = result;
	return;
}

uchar* d_present;
uchar* d_future;

int main() {
	srand(time(NULL));
	on::Tensor<uchar> present = on::rand_ones_and_zeroes(WIDTH, HEIGHT);
	on::Tensor<uchar> future(WIDTH, HEIGHT, (uchar)0);
	array buffer = af::constant(0, WIDTH, HEIGHT);
	array display(WIDTH, HEIGHT, 3);
	af::Window window(WIDTH, HEIGHT);


	bool running = true;
	int start_time = now_ms();

	while (running) {

		int current_time = now_ms();
		int wait_time = (1000 / FPS) - (current_time - start_time);

		on::Kernel::conf_2d(WIDTH, HEIGHT);
		conway_life<<<KERNEL_SHAPE>>>(present, future);
		SYNC_AND_CHECK_FOR_ERRORS(conway_life);
		present = future;

		buffer = (af::array)present;

		gfor(seq i, 3) {
			display(span, span, i) = buffer * 255;
		}

		window.image(display);

		std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));
		start_time = now_ms();
		std::cout << "FPS: " << 1000/wait_time << std::endl;
	}



	return 0;
}
#endif