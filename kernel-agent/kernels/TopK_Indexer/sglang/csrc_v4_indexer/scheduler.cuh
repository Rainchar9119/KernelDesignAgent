#pragma once
#include <deep_gemm/common/math.cuh>

namespace {

using namespace deep_gemm;

// Conditional storage for varlen indices pointer (EBO: zero cost when unused)
template <bool kHasIndices> struct IndicesStorage {
  const uint32_t *indices;
};

template <> struct IndicesStorage<false> {};

template <uint32_t kNextN, bool kIsContextLens2D, bool kIsVarlen,
          uint32_t BLOCK_KV, uint32_t kNumBlocksPerSplit,
          uint32_t kNumNextNAtoms>
struct PagedMQALogitsScheduler : IndicesStorage<kIsVarlen> {
  const uint32_t *context_lens;
  uint32_t batch_size;

  uint32_t current_q_atom_idx, current_kv_idx;
  uint32_t end_q_atom_idx, end_kv_idx;
  uint32_t current_num_kv;
  uint32_t current_advance;
  uint32_t last_advance;

  CUTLASS_DEVICE static uint32_t atom_to_token_idx(const uint32_t &q_atom_idx) {
    if constexpr (kIsVarlen) {
      return q_atom_idx;
    } else {
      static constexpr bool kPadOddN =
          (not kIsVarlen) and (kNextN % 2 == 1) and (kNextN >= 3);
      static constexpr uint32_t kNextNAtom = (kIsVarlen or kNextN >= 2) ? 2 : 1;
      if constexpr (kPadOddN) {
        return q_atom_idx / kNumNextNAtoms * kNextN +
               q_atom_idx % kNumNextNAtoms * kNextNAtom;
      } else {
        return q_atom_idx * kNextNAtom;
      }
    }
  }

  CUTLASS_DEVICE static uint32_t
  atom_to_block_table_row(const uint32_t &q_atom_idx) {
    if constexpr (kIsVarlen) {
      return q_atom_idx;
    } else {
      return q_atom_idx / kNumNextNAtoms;
    }
  }

  CUTLASS_DEVICE uint32_t get_last_advance() const { return last_advance; }

  // NOTES: varlen path reuses a single `is_paired` check to derive both
  // `current_advance` and `current_num_kv`, so `indices` is read only once per
  // atom boundary.
  CUTLASS_DEVICE void refresh_num_kv_and_advance(const uint32_t &q_atom_idx) {
    if constexpr (kIsVarlen) {
      const bool is_paired =
          (q_atom_idx + 1 < batch_size and
           this->indices[q_atom_idx] == this->indices[q_atom_idx + 1]);
      current_advance = is_paired ? 2 : 1;
      const uint32_t ctx_len =
          is_paired ? context_lens[q_atom_idx + 1] : context_lens[q_atom_idx];
      current_num_kv = math::ceil_div(ctx_len, BLOCK_KV);
    } else {
      current_advance = 1;
      const uint32_t q_idx = q_atom_idx / kNumNextNAtoms;
      const auto lens_idx =
          (kIsContextLens2D ? q_idx * kNextN + kNextN - 1 : q_idx);
      current_num_kv = math::ceil_div(context_lens[lens_idx], BLOCK_KV);
    }
  }

  CUTLASS_DEVICE explicit PagedMQALogitsScheduler(const uint32_t &sm_idx,
                                                  const uint32_t &batch_size,
                                                  const uint32_t *context_lens,
                                                  const uint32_t *schedule_meta,
                                                  const uint32_t *indices) {
    this->context_lens = context_lens;
    this->batch_size = batch_size;
    if constexpr (kIsVarlen) {
      this->indices = indices;
    }

    const auto current_pack =
        reinterpret_cast<const uint2 *>(schedule_meta)[sm_idx];
    const auto end_pack =
        reinterpret_cast<const uint2 *>(schedule_meta)[sm_idx + 1];
    current_q_atom_idx = current_pack.x,
    current_kv_idx = current_pack.y * kNumBlocksPerSplit;
    end_q_atom_idx = end_pack.x, end_kv_idx = end_pack.y * kNumBlocksPerSplit;

    // NOTES: unconditional call is safe — reversed metadata allocation ensures
    // `current_q_atom_idx` is always in-bounds.
    refresh_num_kv_and_advance(current_q_atom_idx);
  }

  // Whether num_kv should be refreshed after advancing to q_atom_idx.
  // Varlen: always refresh (each atom may have a different context_len).
  // Non-varlen: only at atom-group boundaries (atoms within a group share
  // context_len).
  CUTLASS_DEVICE bool should_refresh_num_kv(const uint32_t &q_atom_idx) const {
    if constexpr (kIsVarlen) {
      return true;
    } else {
      return q_atom_idx % kNumNextNAtoms == 0;
    }
  }

  CUTLASS_DEVICE bool fetch_next_task(uint32_t &q_atom_idx, uint32_t &kv_idx,
                                      uint32_t &num_kv) {
    q_atom_idx = current_q_atom_idx;
    kv_idx = current_kv_idx;
    num_kv = current_num_kv;
    last_advance = current_advance;

    if (current_q_atom_idx == end_q_atom_idx and current_kv_idx == end_kv_idx)
      return false;

    current_kv_idx += kNumBlocksPerSplit;
    if (current_kv_idx >= current_num_kv) {
      current_kv_idx = 0;
      current_q_atom_idx += current_advance;
      if (should_refresh_num_kv(current_q_atom_idx) and
          exist_q_atom_idx(current_q_atom_idx)) {
        refresh_num_kv_and_advance(current_q_atom_idx);
      }
    }
    return true;
  }

  CUTLASS_DEVICE bool exist_q_atom_idx(const uint32_t &q_atom_idx) const {
    return q_atom_idx < end_q_atom_idx or
           (q_atom_idx == end_q_atom_idx and 0 < end_kv_idx);
  }
};

} // namespace
