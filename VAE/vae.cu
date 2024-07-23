// vae.cu
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

__global__ void encode_kernel(float *input, float *weights1, float *bias1, float *weights2, float *bias2, float *mu, float *logvar, int input_dim, int hidden_dim, int latent_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        float h = 0.0;
        for (int i = 0; i < input_dim; ++i) {
            h += input[i * hidden_dim + idx] * weights1[i * hidden_dim + idx];
        }
        h += bias1[idx];
        h = fmaxf(0.0, h); // ReLU activation

        float mu_val = 0.0;
        float logvar_val = 0.0;
        for (int i = 0; i < hidden_dim; ++i) {
            mu_val += h * weights2[i * latent_dim + idx];
            logvar_val += h * weights2[i * latent_dim + idx];
        }
        mu[idx] = mu_val + bias2[idx];
        logvar[idx] = logvar_val + bias2[idx];
    }
}

__global__ void reparameterize_kernel(float *mu, float *logvar, float *z, int latent_dim, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < latent_dim) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float epsilon = curand_normal(&state);
        z[idx] = mu[idx] + expf(0.5 * logvar[idx]) * epsilon;
    }
}

__global__ void decode_kernel(float *z, float *weights3, float *bias3, float *weights4, float *bias4, float *output, int latent_dim, int hidden_dim, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        float h = 0.0;
        for (int i = 0; i < latent_dim; ++i) {
            h += z[i * hidden_dim + idx] * weights3[i * hidden_dim + idx];
        }
        h += bias3[idx];
        h = fmaxf(0.0, h); // ReLU activation

        for (int i = 0; i < output_dim; ++i) {
            output[i] += h * weights4[idx * output_dim + i];
        }
    }
    __syncthreads();

    int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx2 < output_dim) {
        output[idx2] = 1.0 / (1.0 + expf(-output[idx2])); // Sigmoid activation
    }
}

extern "C" {
    void launch_encode(float *input, float *weights1, float *bias1, float *weights2, float *bias2, float *mu, float *logvar, int input_dim, int hidden_dim, int latent_dim, int batch_size) {
        int threads = 256;
        int blocks = (hidden_dim + threads - 1) / threads;
        encode_kernel<<<blocks, threads>>>(input, weights1, bias1, weights2, bias2, mu, logvar, input_dim, hidden_dim, latent_dim);
    }

    void launch_reparameterize(float *mu, float *logvar, float *z, int latent_dim, unsigned long long seed, int batch_size) {
        int threads = 256;
        int blocks = (latent_dim + threads - 1) / threads;
        reparameterize_kernel<<<blocks, threads>>>(mu, logvar, z, latent_dim, seed);
    }

    void launch_decode(float *z, float *weights3, float *bias3, float *weights4, float *bias4, float *output, int latent_dim, int hidden_dim, int output_dim, int batch_size) {
        int threads = 256;
        int blocks = (hidden_dim + threads - 1) / threads;
        decode_kernel<<<blocks, threads>>>(z, weights3, bias3, weights4, bias4, output, latent_dim, hidden_dim, output_dim);
    }
}
