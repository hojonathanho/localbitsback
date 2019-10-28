#include "ans.h"
#include "gaussian.h"

#include <iostream>
#include <random>

static void fail(const char *assertion, const char *file, int line) {
    std::cerr << file << ':' << line << ": failed check: " << assertion << std::endl;
    std::exit(1);
}
#define CHECK(EX) ((EX) ? (void) 0 : fail(#EX, __FILE__, __LINE__))

void test_discretization() {
    // todo always do this test in the constructor
    Discretization disc(-5, 5, 32);

    auto sym_max = (disc.hi - disc.lo) * (((__uint128_t) 1) << disc.bits) - 1;
    CHECK(sym_max == disc.num_syms() - 1);

    CHECK(disc.real_to_symbol(disc.lo) == 0);
    CHECK(disc.real_to_symbol(disc.hi - 1e-10) == sym_max);

    CHECK(disc.symbol_bin_lo(0) == disc.lo);
    CHECK(disc.symbol_bin_hi(sym_max) == disc.hi);
}

void test_discretized_gaussian(bool pad) {
    DiscretizedGaussian g(
        0.0, 0.1, // mean, std
        Discretization(-5, 5, 12),
        63, // mass_bits
        pad
    );

    auto num_syms = g.disc.num_syms();
    std::vector<uint64_t> cdfs(num_syms), pmfs(num_syms);
    uint64_t cum_pmf = 0;
    for (unsigned int i = 0; i < num_syms; ++i) {
        cdfs[i] = g.cdf(i);
        pmfs[i] = g.pmf(i);
        if (i >= 1) {
            CHECK(cdfs[i] == cum_pmf);
        }
        cum_pmf += pmfs[i];
    }

    auto missing_mass = (long long int) g.total_mass - (long long int) cum_pmf;
    std::cout << "missing mass " << missing_mass << std::endl;
//    std::cout << (double) (missing_mass / g.total_mass) << std::endl;
    CHECK(missing_mass >= 0);

    // test inverse cdf
    DiscretizedGaussian g2(
        0.0, 0.1, // mean, std
        Discretization(-5, 5, 8),
        15, // mass_bits
        pad
    );
    for (uint64_t y = 0; y < g2.total_mass; ++y) {
        symbol_t s = g2.inverse_cdf(y);
        CHECK(g2.cdf(s) <= y && y < g2.cdf(s + 1));
    }
}

void test_ans() {
    ANSBitstream ans(63, 1024, 0);
    size_t init_length = ans.length();
    std::cout << "initial size of bitstream: " << init_length << std::endl;

    // set up probability distribution for coding
    std::vector<double> probs = {0.5, 0.2, 0.3};
    std::vector<uint64_t> pmfs(probs.size()), cdfs(probs.size());
    uint64_t sum_pmfs = 0;
    for (size_t i = 0; i < probs.size(); ++i) {
        pmfs[i] = (uint64_t) ((((__uint128_t) 1) << ans.mass_bits) * probs[i]);
        sum_pmfs += pmfs[i];
    }
    if (sum_pmfs < ((__uint128_t) 1) << ans.mass_bits) {
        pmfs[0] += (((__uint128_t) 1) << ans.mass_bits) - sum_pmfs;
    }
    cdfs[0] = 0;
    for (size_t i = 1; i < probs.size(); ++i) {
        cdfs[i] = cdfs[i - 1] + pmfs[i - 1];
    }

    // Sample from distribution and encode using ANS
    int seed = 0;
    std::mt19937_64 mt(seed);
    std::discrete_distribution<unsigned int> dist({3, 5, 2}); // dist(0, probs.size() - 1);

    int num_samples = 3000;
    std::vector<symbol_t> samples(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        samples[i] = dist(mt);
        ans.encode(pmfs[samples[i]], cdfs[samples[i]]);
    }

    // check bitstream length
    std::cout << "size of bitstream: " << ans.length() << std::endl;
    double net_bitrate = (double) (ans.length() - init_length) / num_samples;
    std::cout << "net bitrate: " << net_bitrate << std::endl;

    double expected_bitrate = 0; // calculate cross entropy
    auto dist_probabilities = dist.probabilities();
    for (size_t i = 0; i < dist_probabilities.size(); ++i) {
        expected_bitrate -= dist_probabilities[i] * log2(probs[i]);
    }
    std::cout << "expected bitrate: " << expected_bitrate << " deviation: " << abs(net_bitrate - expected_bitrate)  << std::endl;
    CHECK(abs(net_bitrate - expected_bitrate) < 0.003);

    // decode and check that we get the same thing
    for (int i = 0; i < num_samples; ++i) {
        auto p = ans.peek();

        // inverse cdf to get the symbol
        symbol_t s = 0;
        for ( ; s < cdfs.size() - 1; ++s) {
            if (cdfs[s] <= p && p < cdfs[s + 1]) {
                break;
            }
        }

        CHECK(s == samples[num_samples - i - 1]);
        ans.decode(p, pmfs[s], cdfs[s]);
    }

    std::cout << "final size of bitstream: " << ans.length() << std::endl;
    CHECK(ans.length() == init_length);
}

void test_ans_gaussian() {
    int ans_precision = 63;
    int disc_precision = 32;
    ANSBitstream ans(ans_precision, 1024, 0);
    size_t init_length = ans.length();
    DiscretizedGaussian g(1.9 /* mean */, 1e-6 /* std */, Discretization(-5, 5, disc_precision), ans_precision, true /* pad */);

    int seed = 0;
    std::mt19937_64 mt(seed);
    std::normal_distribution<double> dist(g.mean, g.std);

    int num_samples = 3000 * 10;
    std::vector<double> samples(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        samples[i] = g.disc.round(dist(mt)); // encode rounded samples; should get perfect decoding
        symbol_t sym = g.disc.real_to_symbol(samples[i]);
        ans.encode(g.pmf(sym), g.cdf(sym));
    }
    double net_bitrate = (double) (ans.length() - init_length) / num_samples;
    double estimated_diffent = net_bitrate - disc_precision;
    std::cout << "estimated differential entropy: " << estimated_diffent << std::endl;
    double expected_diffent = log2((double) g.std * sqrt(2.0 * M_PI * M_E));
    std::cout << "expected: " << expected_diffent << ' ' << abs(expected_diffent - estimated_diffent) << std::endl;
    CHECK(abs(expected_diffent - estimated_diffent) < 0.01);

    // check decoding
    for (int i = 0; i < num_samples; ++i) {
        auto p = ans.peek();
        symbol_t sym = g.inverse_cdf(p);
        double sample = g.disc.symbol_to_real(sym);
        CHECK(sample == samples[num_samples - i - 1]);
        ans.decode(p, g.pmf(sym), g.cdf(sym));
    }

    std::cout << "final size of bitstream: " << ans.length() << std::endl;
    CHECK(ans.length() == init_length);
}

void run_tests() {
    test_discretization();
    test_discretized_gaussian(true /* pad */);
    test_discretized_gaussian(false /* pad */);
    test_ans();
    test_ans_gaussian();
    std::cout << "all tests passed" << std::endl;
}
