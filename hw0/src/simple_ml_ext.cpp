#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace py = pybind11;

void matmul(float* out, const float* m1, const float* m2, size_t m, size_t n, size_t k) {
  for (size_t i = 0; i < m; i ++) {
    for (size_t j = 0; j < k; j ++) {
      out[k*i + j] = 0;
      for (size_t t = 0; t < n; t ++) {
        out[k * i + j] += (m1[i * n + t] * m2[t * k + j]);
      }
    }
  }
}

void transpose(float* out, const float* m1, size_t m, size_t n) {
  for (size_t i = 0; i < n; i ++) {
    for (size_t j = 0; j < m; j ++) {
      out[i * m + j] = m1[j * n + i];
    }
  }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    float* X_batch = new float[batch * n];
    unsigned char* y_batch = new unsigned char[batch];
    float* scores = new float[batch * k];
    float* Iy = new float[batch * k];
    float* gradient = new float[n * k];
    float* transpose_buffer = new float[batch * n];

    for (size_t i = 0; i < m; i += batch) {
        size_t current_bs = std::min(m - i, batch);

        for (size_t cnt = 0; cnt < current_bs; cnt++) {
            memcpy(X_batch + cnt * n, X + (i + cnt) * n, sizeof(float) * n);
            y_batch[cnt] = y[i + cnt];
        }

        matmul(scores, X_batch, theta, current_bs, n, k);

        // Ensure numerical stability by subtracting the max logits
        for (size_t row = 0; row < current_bs; row++) {
            float row_max = scores[row * k];
            for (size_t col = 1; col < k; col++) {
                if (scores[row * k + col] > row_max) {
                    row_max = scores[row * k + col];
                }
            }
            for (size_t col = 0; col < k; col++) {
                scores[row * k + col] -= row_max;
            }
        }

        // Apply softmax normalization
        for (size_t row = 0; row < current_bs; row++) {
            float row_sum = 0.0f;
            for (size_t col = 0; col < k; col++) {
                scores[row * k + col] = exp(scores[row * k + col]);
                row_sum += scores[row * k + col];
            }
            for (size_t col = 0; col < k; col++) {
                scores[row * k + col] /= row_sum;
            }
        }

        memset(Iy, 0, current_bs * k * sizeof(float));
        for (size_t row = 0; row < current_bs; row++) {
            Iy[row * k + y_batch[row]] = 1.0f;
        }

        for (size_t row = 0; row < current_bs; row++) {
            for (size_t col = 0; col < k; col++) {
                scores[row * k + col] -= Iy[row * k + col];
            }
        }

        transpose(transpose_buffer, X_batch, current_bs, n);
        matmul(gradient, transpose_buffer, scores, n, current_bs, k);

        for (size_t row = 0; row < n; row++) {
            for (size_t col = 0; col < k; col++) {
                theta[row * k + col] -= (lr * gradient[row * k + col] / current_bs);
            }
        }
    }

    delete[] X_batch;
    delete[] y_batch;
    delete[] scores;
    delete[] Iy;
    delete[] gradient;
    delete[] transpose_buffer;
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
