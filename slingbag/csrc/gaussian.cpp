#include <torch/extension.h>
#include <vector>

torch::Tensor gaussian_cuda_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    double, int64_t, double, double, int64_t);
std::vector<torch::Tensor> gaussian_cuda_backward(torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, double, int64_t, double, double, bool);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gaussian_cuda_forward);
    m.def("backward", &gaussian_cuda_backward);
}
