#pragma once

#include <cassert>
#include <iostream>

typedef uint64_t symbol_t;

struct Discretization {
    int lo, hi; // endpoints of discretization range
    int bits; // bits of precision

    Discretization(int lo_, int hi_, int bits_) : lo(lo_), hi(hi_), bits(bits_) {
        assert(hi > lo);
        // check overflow of symbol_t?
    }

    symbol_t num_syms() const {
        return ((symbol_t) (hi - lo)) << bits; // number of bins
    }

    symbol_t real_to_symbol(double x) const {
        assert(lo <= x && x < hi);
        // lo -> 0, hi-epsilon -> self.num_syms-1
        return ldexp((double) x - lo, bits); // (x - range_lo) << disc_bits
    }

    double symbol_to_real(symbol_t s) const {
        assert(s < num_syms());
        return ldexp((double) s + 0.5, -bits) + lo;  // midpoint of bin
    }

    double symbol_bin_lo(symbol_t s) const {
        assert(s < num_syms());
        return ldexp((double) s, -bits) + lo;
    }

    double symbol_bin_hi(symbol_t s) const {
        assert(s < num_syms());
        return ldexp((double) s + 1.0, -bits) + lo;
    }

    double round(double x) const {
        return symbol_to_real(real_to_symbol(x));
    }
};


// CDF of normal distribution
inline long double normalcdf(long double x, long double mean, long double std) {
    return 0.5 * (1.0 + erfl((x - mean) * M_SQRT1_2 / std));
}

inline uint64_t scaled_normalcdf_diff(long double x0, long double x1, long double mean, long double std, uint64_t prob_scale) {
    uint64_t u0 = (uint64_t) (prob_scale * normalcdf(x0, mean, std));
    uint64_t u1 = (uint64_t) (prob_scale * normalcdf(x1, mean, std));
    if (u0 < u1) {
        throw std::runtime_error("invalid CDF calculation");
    }
    return u0 - u1;
}


struct DiscretizedGaussian {
    double mean, std;
    Discretization disc;
    bool pad;
    uint64_t total_mass;
    uint64_t prob_scale;

    DiscretizedGaussian(double mean_, double std_, const Discretization& disc_, int mass_bits_ /* log mass */, bool pad_)
            : mean(mean_), std(std_), disc(disc_), pad(pad_),
              total_mass(((uint64_t) 1) << mass_bits_),
              prob_scale(pad_ ? (total_mass - disc.num_syms()) : total_mass)
    {
        if (!(0 <= mass_bits_ && mass_bits_ <= 63)) {
            throw std::runtime_error("expected mass_bits in [0, 63]");
        }
        if (pad && (total_mass <= disc.num_syms())) { // i.e. assert prob_scale >= 1
            throw std::runtime_error("since padding is enabled, expected total_mass > num_syms");
        }
        //assert(((uint128_t) prob_scale) <= total_mass - disc.num_syms()); // make sure that prob_scale is rounded down
    }

    uint64_t pmf(symbol_t s) {
        auto prob = scaled_normalcdf_diff(disc.symbol_bin_hi(s), disc.symbol_bin_lo(s), mean, std, prob_scale);
        return pad ? (prob + 1) : prob;
    }

    uint64_t cdf(symbol_t s) {
        if (s == 0) {
            return 0;
        }
        auto prob = scaled_normalcdf_diff(disc.symbol_bin_hi(s - 1), disc.symbol_bin_lo(0), mean, std, prob_scale);
        return pad ? (prob + s) : prob;
    }

    symbol_t inverse_cdf(uint64_t y) {
        // Finds symbol s such that cdf(s) <= y < cdf(s+1)
        assert(y < total_mass);
        symbol_t lo = 0;
        symbol_t hi = disc.num_syms();
        while (lo < hi) {
            symbol_t mid = lo + (hi - lo) / 2;
            if (y < cdf(mid)) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        auto s = lo - 1;
        if (!(cdf(s) <= y && y < cdf(s + 1))) {
            std::cout << "expected " << cdf(s) << " <= " << y << " < " << cdf(s+1) << std::endl;
            dump(s);
            throw std::runtime_error("inverse cdf failed (could be a gaussian cdf overflow problem)");
        }
        if (s >= disc.num_syms()) {
            dump(s);
            throw std::runtime_error("inverse cdf returned out of range symbol");
        }

        return lo - 1;
    }

private:
    void dump(symbol_t s) {
        std::cout << "\ts=" << s << " hi=" << disc.num_syms() << " real(s) " << disc.symbol_bin_lo(s) << "~~~" << disc.symbol_bin_hi(s) << std::endl;
        std::cout << "\tmean=" << mean << " std=" << std << " prob_scale=" << prob_scale << std::endl;
        std::cout << "\ttotal_mass=" << total_mass << std::endl;
    }
};

