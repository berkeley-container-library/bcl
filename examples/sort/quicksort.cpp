#include <cstdlib>
#include <cstdio>

#include <BCL.hpp>
#include <containers/CircularQueue.hpp>

std::vector <int> random_ints(int n, int range) {
  std::vector <int> ri(n);
  for (int i = 0; i < ri.size(); i++) {
    ri[i] = lrand48() % range;
  }
  return ri;
}

// Pick pivot by looking at sample random samples per
// node and picking the best pivot
int pick_random_sample_pivot(const std::vector <int> &data, int samples) {
  std::vector <int> random_samples(samples);
  for (int i = 0; i < samples; i++) {
    random_samples[i] = data[lrand48() % data.size()];
  }
  // Sort is overkill here; we really want quickselect
  std::sort(random_samples.begin(), random_samples.end());
  int my_pivot = random_samples[samples / 2];
  int pivot = BCL::allreduce(my_pivot, BCL::plus <int> {});
  pivot = pivot / BCL::nprocs();
  return pivot;
}

void redistribute(std::vector <int> &data,
  std::vector <BCL::CircularQueue <int>> &queues, int pivot) {
  std::vector <std::vector <int>> buffers;
  for (int rank = 0; rank < BCL::nprocs(); rank++) {
    buffers.push_back(std::vector <int> ());
  }

  int max_buffer = 200;

  int pivot_node = BLC::nprocs() / 2;

  for (const int &datum : data) {
    if (datum < pivot) {
    }
  }
}

int main(int argc, char **argv) {
  BCL::init();

  int n_data = 1000;
  int n_random_samples = 10;

  srand48(BCL::rank());

  int queue_size = n_data * 3;

  std::vector <BCL::CircularQueue <int>> queues;

  for (int rank = 0; rank < BCL::nprocs(); rank++) {
    queues.push_back(BCL::CircularQueue <int> (rank, queue_size));
  }

  std::vector <int> data = random_ints(n_data, 100000);

  int pivot = pick_random_sample_pivot(data, n_random_samples);
  printf("%d got pivot %d\n", BCL::rank(), pivot);

  BCL::finalize();
  return 0;
}
