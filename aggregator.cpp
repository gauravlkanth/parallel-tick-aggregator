#include <math.h>

#include <aggregate.hpp>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>
#define NUM_SYMBOLS 1024

// WHAT WORKED BIG TIME FOR ME
// alignas(64) to prevent false sharing between threads
// __restrict__ to say to the compiler that two mem regions don't overlap
// since the aggregator is memory bound and not compute bound, __builtin_prefetch did wonders
// also #pragma GCC unroll N, unrolls the loop N times, that is it groups computations into groups of 8
// combine that with SIMD instructions one gets with -O3 compiler flag,
// the loop time is cut down by almost N times
// increasing throughput by a massive amount

// also for some reason if i didn't memset(out, 0) i got UB, don't know why tho
// should'nt the judge be responsible for a clear and clean out buffer !!

class FastAggregator final : public csot::Aggregator {
   public:
    void on_init(std::uint32_t p_numsymbols) override {
        partials.assign(4, PartialTable{});
    }

    void run(const csot::AggTick* ticks, std::size_t n, csot::SymbolAgg* out) override {
        std::memset(out, 0, NUM_SYMBOLS * sizeof(csot::SymbolAgg));

        const uint32_t T = 4;
        std::vector<std::jthread> ws;
        for (uint32_t k = 0; k < T; ++k) {
            const size_t lo = n * k / T;
            const size_t hi = n * (k + 1) / T;
            ws.emplace_back([=]() { reduce_chunk(ticks + lo, hi - lo, partials[k].row); });
        }
        ws.clear();

        // merge
        for (size_t k = 0; k < T; ++k) {
            for (size_t s = 0; s < NUM_SYMBOLS; ++s) {
                if (partials[k].row[s].count == 0) continue;

                if (out[s].count == 0) {
                    out[s] = partials[k].row[s];
                } else {
                    out[s].count += partials[k].row[s].count;
                    out[s].sum_price += partials[k].row[s].sum_price;
                    out[s].sum_qty += partials[k].row[s].sum_qty;

                    if (partials[k].row[s].max_price > out[s].max_price)
                        out[s].max_price = partials[k].row[s].max_price;

                    if (partials[k].row[s].min_price < out[s].min_price)
                        out[s].min_price = partials[k].row[s].min_price;
                }
            }
        }
    }

   private:
    struct alignas(64) PartialTable {
        csot::SymbolAgg row[NUM_SYMBOLS];
    };
    void reduce_chunk(const csot::AggTick* __restrict__ start, size_t n, csot::SymbolAgg* __restrict__ row) {
        for (size_t s = 0; s < NUM_SYMBOLS; ++s) {
            row[s].count = 0;
            row[s].sum_price = 0;
            row[s].sum_qty = 0;
            row[s].max_price = 0;
            row[s].min_price = std::numeric_limits<decltype(row[s].min_price)>::max();
        }
#pragma GCC unroll 8
        for (size_t i = 0; i < n; ++i) {
            __builtin_prefetch(&start[i + 64], 0, 3);
            const csot::AggTick& tick = start[i];
            csot::SymbolAgg& agg = row[tick.symbol_id];

            if (tick.price > agg.max_price) agg.max_price = tick.price;
            if (tick.price < agg.min_price) agg.min_price = tick.price;

            ++agg.count;
            agg.sum_price += tick.price;
            agg.sum_qty += tick.qty;
        }
    }

    std::vector<PartialTable> partials;
};

extern "C" csot::Aggregator* create_aggregator() {
    return new FastAggregator();
}
