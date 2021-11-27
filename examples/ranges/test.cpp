#include <vector>
#include <span>

int main(int argc, char** argv) {
  std::vector<int> v{10, 12, 13};
  const std::vector<int>& vr = v;

  std::span<const int> span(vr);

  return 0;
}
