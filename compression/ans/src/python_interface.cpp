#include "ans.h"
#include "gaussian.h"

#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;

using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;
typedef Eigen::Array<symbol_t, Eigen::Dynamic, 1> ArrayXu;
typedef Eigen::Array<symbol_t, Eigen::Dynamic, Eigen::Dynamic> ArrayXXu;


class PyANS {
public:
    PyANS(int ans_mass_bits, int ans_init_bits, int ans_init_seed, int num_streams) {
        for (int i = 0; i < num_streams; ++i) {
            m_streams.emplace_back(ans_mass_bits, ans_init_bits, ans_init_seed + i);
        }
    }

    PyANS(py::list py_list) {
        // Construct straight from saved Python format
        from_py(py_list);
    }

    py::list to_py() {
        // Save all streams to a Python list
        assert_some_streams();
        py::list py_list;
        for (auto& st : m_streams) {
            // Store the stream state as a dictionary
            py_list.append(py::dict(
                "stream"_a=py::bytes((char*) &st.stream[0], st.stream.size() * sizeof(uint64_t)),
                "tip_lo"_a=(uint64_t) st.tip,
                "tip_hi"_a=(uint64_t) (st.tip >> 64),
                "mass_bits"_a=st.mass_bits
            ));
        }
        return py_list;
    }

    void from_py(py::list py_list) {
        // Load all streams from the saved Python format.
        // Note: this clears the current streams, and possibly changes the number of streams.
        m_streams.clear();
        for (auto dict : py_list) {
            // Create a stream with the appropriate setting of mass_bits
            m_streams.emplace_back(dict["mass_bits"].cast<int>(), 0, 0);
            auto& st = m_streams.back();
            // Load the tip data
            st.tip = (((uint128_t) dict["tip_hi"].cast<uint64_t>()) << 64)
                     | ((uint128_t) dict["tip_lo"].cast<uint64_t>());
            // Load the stream vector
            py::bytes py_stream = dict["stream"].cast<py::bytes>();
            char *buffer;
            ssize_t length;
            if (PYBIND11_BYTES_AS_STRING_AND_SIZE(py_stream.ptr(), &buffer, &length)) {
                throw std::runtime_error("Unable to extract bytes contents!");
            }
            if ((length % sizeof(uint64_t)) != 0) {
                throw std::runtime_error("Invalid stored stream length. Must be a sequence of 64-bit numbers.");
            }
            st.stream.resize(length / sizeof(uint64_t));
            memcpy(&st.stream[0], buffer, length);
        }
    }


    size_t stream_length() {
        size_t total = 0;
        for (const auto& ans : m_streams) {
            total += ans.length();
        }
        return total;
    }

    void encode_gaussian_diag(Eigen::Ref<ArrayXu> x, Eigen::Ref<VectorXd> means, Eigen::Ref<VectorXd> stds, const Discretization& disc, bool pad) {
        assert_some_streams();

        int dim = x.size();
        if (means.size() != dim || stds.size() != dim) {
            throw std::runtime_error("mismatched dimension");
        }
        if ((stds.array() < 0.).any()) {
            throw std::runtime_error("negative standard deviation");
        }
        parallelize(dim, [&](ANSBitstream& ans, int start, int end) {
            // encode backwards
            for (int i = end - 1; i >= start; --i) {
                DiscretizedGaussian g(means(i), stds(i), disc, ans.mass_bits, pad);
                symbol_t sym = x(i);
                ans.encode(g.pmf(sym), g.cdf(sym));
            }
        });
    }

    ArrayXu decode_gaussian_diag(Eigen::Ref<VectorXd> means, Eigen::Ref<VectorXd> stds, const Discretization& disc, bool pad) {
        assert_some_streams();

        int dim = means.size();
        if (stds.size() != dim) {
            throw std::runtime_error("mismatched dimension");
        }
        if ((stds.array() < 0.).any()) {
            throw std::runtime_error("negative standard deviation");
        }

        ArrayXu out(dim);
        parallelize(dim, [&](ANSBitstream& ans, int start, int end) {
            // decode forward
            for (int i = start; i < end; ++i) {
                DiscretizedGaussian g(means(i), stds(i), disc, ans.mass_bits, pad);
                auto p = ans.peek();
                symbol_t sym = g.inverse_cdf(p);
                out(i) = sym;
                ans.decode(p, g.pmf(sym), g.cdf(sym));
            }
        });
        return out;
    }

    void encode_gaussian_single(Eigen::Ref<ArrayXu> x, py::EigenDRef<MatrixXd> mean_coefs,
                                Eigen::Ref<VectorXd> biases, Eigen::Ref<VectorXd> stds,
                                bool left_to_right, const Discretization& disc, bool pad) {
        assert_some_streams();

        // single datapoint (vector from a multivariate Gaussian): always use stream 0
        generic_encode_gaussian(m_streams[0], x, mean_coefs, biases, stds, left_to_right, disc, pad);
    }

    ArrayXu decode_gaussian_single(py::EigenDRef<MatrixXd> mean_coefs,
                                   Eigen::Ref<VectorXd> biases, Eigen::Ref<VectorXd> stds,
                                   bool left_to_right, const Discretization& disc, bool pad) {
        assert_some_streams();

        int dim = stds.size();
        ArrayXu x(dim);
        // single datapoint (vector from a multivariate Gaussian): always use stream 0
        generic_decode_gaussian(m_streams[0], x, mean_coefs, biases, stds, left_to_right, disc, pad);
        return x;
    }

    void encode_gaussian_batched(py::EigenDRef<ArrayXXu> xs, py::EigenDRef<MatrixXd> mean_coefs,
                                 py::EigenDRef<MatrixXd> biases, Eigen::Ref<VectorXd> stds,
                                 bool left_to_right, const Discretization& disc, bool pad) {
        assert_some_streams();

        if (biases.rows() != xs.rows()) {
            throw std::runtime_error("inconsistent batch size for data and biases");
        }
        parallelize(xs.rows(), [&](ANSBitstream& ans, int start, int end) {
            for (int i = end - 1; i >= start; --i) {
                generic_encode_gaussian(ans, xs.row(i), mean_coefs, biases.row(i).transpose(), stds, left_to_right, disc, pad);
            }
        });
    }

    ArrayXXu decode_gaussian_batched(py::EigenDRef<MatrixXd> mean_coefs, /* (d,d) */
                                     py::EigenDRef<MatrixXd> biases, /* (n,d) */
                                     Eigen::Ref<VectorXd> stds, /* (d,1) */
                                     bool left_to_right, const Discretization& disc, bool pad) {
        assert_some_streams();

        int count = biases.rows();
        int dim = stds.size();
        ArrayXXu xs(count, dim);
        parallelize(count, [&](ANSBitstream& ans, int start, int end) {
            for (int i = start; i < end; ++i) {
                generic_decode_gaussian(ans, xs.row(i), mean_coefs, biases.row(i).transpose(), stds, left_to_right, disc, pad);
            }
        });
        return xs;
    }

private:
    std::vector<ANSBitstream> m_streams;

    void assert_some_streams() {
        if (m_streams.empty()) {
            throw std::runtime_error("No ANS streams present");
        }
    }

    // Distribute ANS streams over threads
    void parallelize(int total, const std::function<void(ANSBitstream&, int, int)>& f) {
        #pragma omp parallel for schedule(runtime)
        for (int i_stream = 0; i_stream < m_streams.size(); ++i_stream) {
            // Each stream takes care of a block of data
            int extra = total % m_streams.size();
            int start = (total / m_streams.size()) * i_stream + std::min(i_stream, extra);
            int end = start + (total / m_streams.size()) + (int) (i_stream < extra);
            f(m_streams[i_stream], start, end);
        }
    }

    // Calculates the mean of one component of a multivariate Gaussian arranged in a linear autoregressive fashion
    template<typename A, typename B, typename C>
    double calc_mean(const Eigen::DenseBase<A>& x, const Eigen::DenseBase<B>& mean_coefs, const Eigen::DenseBase<C>& biases,
                     int pos, int dim, bool left_to_right) {
        if (left_to_right) {
            // left-to-right sampling: mean_coefs is lower triangular
            if (pos == 0) { return biases(pos); }
            // return mean_coefs.row(pos).head(pos).dot(x.head(pos) - biases.head(pos)) + biases(pos); // note don't use this for determinism?
            double out = biases(pos);
            for (int i = 0; i < pos; ++i) {
                out += mean_coefs(pos,i) * (x(i) - biases(i));
            }
            return out;
        } else {
            // right-to-left sampling: mean_coefs is upper triangular
            if (pos == dim - 1) { return biases(pos); }
            // return mean_coefs.row(pos).tail(dim - pos - 1).dot(x.tail(dim - pos - 1) - biases.tail(dim - pos - 1)) + biases(pos);
            double out = biases(pos);
            for (int i = pos - 1; i < dim; ++i) {
                out += mean_coefs(pos,i) * (x(i) - biases(i));
            }
            return out;
        }
    }

    // Encoding for a multivariate Gaussian specified as a linear AR model
    template<typename A, typename B, typename C, typename D>
    void generic_encode_gaussian(ANSBitstream& ans,
                                 const Eigen::DenseBase<A>& x,
                                 const Eigen::DenseBase<B>& mean_coefs,
                                 const Eigen::DenseBase<C>& biases,
                                 const Eigen::MatrixBase<D>& stds,
                                 bool left_to_right, const Discretization& disc, bool pad) {
        int dim = x.size();
        if (mean_coefs.rows() != dim || mean_coefs.cols() != dim) {
            throw std::runtime_error("mismatched size for mean_coefs");
        }
        if (biases.size() != dim) {
            throw std::runtime_error("mismatched size for biases");
        }
        if (stds.size() != dim) {
            throw std::runtime_error("mismatched size for stds");
        }
        // TODO stricter checks here for rows & cols of everything
        if ((stds.array() < 0.).any()) {
            throw std::runtime_error("negative standard deviation");
        }

        VectorXd x_real(dim);
        for (int i = 0; i < dim; ++i) { x_real(i) = disc.symbol_to_real(x(i)); }

        for (int i = 0; i < dim; ++i) {
            int pos = left_to_right ? (dim - i - 1) : i; // encode backwards
            DiscretizedGaussian g(calc_mean(x_real, mean_coefs, biases, pos, dim, left_to_right), stds(pos), disc, ans.mass_bits, pad);
            symbol_t sym = x(pos);
            ans.encode(g.pmf(sym), g.cdf(sym));
        }
    }

    // Decoding for a multivariate Gaussian specified as a linear AR model
    template<typename A, typename B, typename C, typename D>
    void generic_decode_gaussian(ANSBitstream& ans,
                                 Eigen::DenseBase<A> const & out_x,
                                 const Eigen::DenseBase<B>& mean_coefs,
                                 const Eigen::DenseBase<C>& biases,
                                 const Eigen::MatrixBase<D>& stds,
                                 bool left_to_right, const Discretization& disc, bool pad) {
        int dim = stds.size();
        if (mean_coefs.rows() != dim || mean_coefs.cols() != dim) {
            throw std::runtime_error("mismatched size for mean_coefs");
        }
        if (biases.size() != dim) {
            throw std::runtime_error("mismatched size for biases");
        }
        if ((stds.array() < 0.).any()) {
            throw std::runtime_error("negative standard deviation");
        }

        VectorXd x_real = VectorXd::Zero(dim);
        for (int i = 0; i < dim; ++i) {
            int pos = left_to_right ? i : (dim - i - 1); // decode forwards
            DiscretizedGaussian g(calc_mean(x_real, mean_coefs, biases, pos, dim, left_to_right), stds(pos), disc, ans.mass_bits, pad);
            auto p = ans.peek();
            symbol_t sym = g.inverse_cdf(p);
            const_cast<Eigen::DenseBase<A>&>(out_x)(pos) = sym; // https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
            x_real(pos) = disc.symbol_to_real(sym);
            ans.decode(p, g.pmf(sym), g.cdf(sym));
        }
    }
};


PYBIND11_MODULE(fast_ans, m) {
    extern void run_tests();
    m.def("run_tests", &run_tests);

    py::class_<Discretization>(m, "Discretization")
        .def(py::init<int, int, int>(), "lo"_a, "hi"_a, "bits"_a)
        .def("real_to_symbol", py::vectorize(&Discretization::real_to_symbol))
        .def("symbol_to_real", py::vectorize(&Discretization::symbol_to_real))
    ;

    py::class_<PyANS>(m, "ANS")
        .def(py::init<int, int, int, int>(), "ans_mass_bits"_a, "ans_init_bits"_a, "ans_init_seed"_a, "num_streams"_a)
        .def(py::init<py::list>())
        .def("stream_length", &PyANS::stream_length)
        .def("to_py", &PyANS::to_py)
        .def("from_py", &PyANS::from_py)
        .def("encode_gaussian_diag", &PyANS::encode_gaussian_diag, "x"_a, "means"_a, "stds"_a, "disc"_a, "pad"_a)
        .def("decode_gaussian_diag", &PyANS::decode_gaussian_diag, "means"_a, "stds"_a, "disc"_a, "pad"_a)
        .def("encode_gaussian", &PyANS::encode_gaussian_single, "x"_a, "mean_coefs"_a, "biases"_a, "stds"_a, "left_to_right"_a, "disc"_a, "pad"_a)
        .def("decode_gaussian", &PyANS::decode_gaussian_single, "mean_coefs"_a, "biases"_a, "stds"_a, "left_to_right"_a, "disc"_a, "pad"_a)
        .def("encode_gaussian_batched", &PyANS::encode_gaussian_batched, "xs"_a, "mean_coefs"_a, "biases"_a, "stds"_a, "left_to_right"_a, "disc"_a, "pad"_a)
        .def("decode_gaussian_batched", &PyANS::decode_gaussian_batched, "mean_coefs"_a, "biases"_a, "stds"_a, "left_to_right"_a, "disc"_a, "pad"_a)
    ;
}
