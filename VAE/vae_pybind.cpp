// vae_pybind.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern "C" {
    void launch_encode(float *input, float *weights1, float *bias1, float *weights2, float *bias2, float *mu, float *logvar, int input_dim, int hidden_dim, int latent_dim, int batch_size);
    void launch_reparameterize(float *mu, float *logvar, float *z, int latent_dim, unsigned long long seed, int batch_size);
    void launch_decode(float *z, float *weights3, float *bias3, float *weights4, float *bias4, float *output, int latent_dim, int hidden_dim, int output_dim, int batch_size);
}

void py_encode(py::array_t<float> input, py::array_t<float> weights1, py::array_t<float> bias1, py::array_t<float> weights2, py::array_t<float> bias2, py::array_t<float> mu, py::array_t<float> logvar, int input_dim, int hidden_dim, int latent_dim, int batch_size) {
    launch_encode(input.mutable_data(), weights1.mutable_data(), bias1.mutable_data(), weights2.mutable_data(), bias2.mutable_data(), mu.mutable_data(), logvar.mutable_data(), input_dim, hidden_dim, latent_dim, batch_size);
}

void py_reparameterize(py::array_t<float> mu, py::array_t<float> logvar, py::array_t<float> z, int latent_dim, unsigned long long seed, int batch_size) {
    launch_reparameterize(mu.mutable_data(), logvar.mutable_data(), z.mutable_data(), latent_dim, seed, batch_size);
}

void py_decode(py::array_t<float> z, py::array_t<float> weights3, py::array_t<float> bias3, py::array_t<float> weights4, py::array_t<float> bias4, py::array_t<float> output, int latent_dim, int hidden_dim, int output_dim, int batch_size) {
    launch_decode(z.mutable_data(), weights3.mutable_data(), bias3.mutable_data(), weights4.mutable_data(), bias4.mutable_data(), output.mutable_data(), latent_dim, hidden_dim, output_dim, batch_size);
}

PYBIND11_MODULE(vae, m) {
    m.def("encode", &py_encode, "Encode function");
    m.def("reparameterize", &py_reparameterize, "Reparameterize function");
    m.def("decode", &py_decode, "Decode function");
}
