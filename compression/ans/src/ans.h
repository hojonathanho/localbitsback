#pragma once

// Based on https://github.com/rygorous/ryg_rans/blob/master/rans64.h

#include <random>
#include <stdexcept>
#include <vector>
#include <iostream>

typedef __uint128_t uint128_t;
constexpr uint128_t ONE = 1;
constexpr uint128_t RANS_L = ONE << 63;

struct ANSBitstream {
    std::vector<uint64_t> stream;
    uint128_t tip;
    int mass_bits; // distribution masses assumed to sum to 1<<mass_bits

    ANSBitstream(int mass_bits_, int init_bits_, int init_seed_) : tip(RANS_L), mass_bits(mass_bits_) {
        if (mass_bits < 1 || mass_bits > 63) {
            throw std::runtime_error("mass_bits must be in [1, 63]");
        }
        if (init_bits_ > 0) {
            // Random init bits from a uniform distribution
            std::mt19937_64 mt(init_seed_);
            std::uniform_int_distribution<uint64_t> dist(0, (ONE << mass_bits) - 1);
            for (int i = 0; i < (init_bits_ + mass_bits - 1) / mass_bits; ++i) {
                encode(1, dist(mt));
            }
        }
    }

    void encode(uint64_t pmf, uint64_t cdf) {
        if (tip >= ((RANS_L >> mass_bits) << 64) * pmf) {
            stream.push_back(tip);
            tip >>= 64;
        }
        tip = ((tip / pmf) << mass_bits) + (tip % pmf) + cdf;
    }

    void decode(uint64_t peeked, uint64_t pmf, uint64_t cdf) {
        tip = pmf * (tip >> mass_bits) + peeked - cdf;
        if (tip < RANS_L) {
            if (stream.empty()) {
                throw std::runtime_error("Empty bitstream!");
            }
            tip = (tip << 64) | stream.back();
            stream.pop_back();
        }
    }

    uint64_t peek() const {
        return tip & ((ONE << mass_bits) - 1);
    }

    size_t tip_length() const {
        // count bits in the tip
        size_t size = 0;
        uint128_t tip_copy = tip;
        while (tip_copy != 0) {
            ++size;
            tip_copy >>= 1;
        }
        return size;
    }

    size_t length() const { // length in bits
        return 64 * stream.size() + tip_length();
    }
};
