"""
Tests for PagedKVCacheManager
==============================

Validates correctness of allocation, append, fork/CoW, free,
packed KV building, and sliding window trimming.

Run with: python test_paged_kv_cache.py
"""

import torch
import sys

from paged_kv_cache import PagedKVCacheManager, BlockAllocator


# ---- helpers ----

def make_mgr(**overrides) -> PagedKVCacheManager:
    """Create a small manager for testing."""
    defaults = dict(
        num_layers=2,
        num_kv_heads=4,
        head_dim=8,
        max_blocks=64,
        block_size=4,
        device="cpu",
        dtype=torch.float32,
    )
    defaults.update(overrides)
    return PagedKVCacheManager(**defaults)


def rand_kv(num_tokens, num_kv_heads=4, head_dim=8):
    """Generate random K, V tensors."""
    k = torch.randn(num_tokens, num_kv_heads, head_dim)
    v = torch.randn(num_tokens, num_kv_heads, head_dim)
    return k, v


def append_all_layers(mgr, seq_id, k, v):
    """Append the same k,v to all layers (simulating a full forward pass)."""
    for layer_idx in range(mgr.num_layers):
        mgr.append_tokens(seq_id, layer_idx, k, v)


passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


# ============================================================================
# TEST 1: Block Allocator basics
# ============================================================================
print("\n=== Test 1: BlockAllocator ===")

alloc = BlockAllocator(8)
check("initial free count", alloc.num_free == 8)
check("initial used count", alloc.num_used == 0)

b0 = alloc.alloc()
check("alloc returns valid index", 0 <= b0 < 8)
check("used count after alloc", alloc.num_used == 1)
check("not shared initially", not alloc.is_shared(b0))

alloc.ref(b0)
check("shared after ref", alloc.is_shared(b0))
check("refcount is 2", alloc.refcounts[b0] == 2)

alloc.release(b0)
check("not shared after one release", not alloc.is_shared(b0))
check("still used (refcount=1)", alloc.num_used == 1)

alloc.release(b0)
check("freed after second release", alloc.num_free == 8)


# ============================================================================
# TEST 2: Basic sequence lifecycle
# ============================================================================
print("\n=== Test 2: Sequence lifecycle ===")

mgr = make_mgr()
sid = mgr.allocate_sequence()
check("seq allocated", sid == 0)
check("initial length = 0", mgr.get_kv_length(sid) == 0)

# Append 3 tokens (fits in one block of size 4)
k, v = rand_kv(3)
append_all_layers(mgr, sid, k, v)
check("length after append(3)", mgr.get_kv_length(sid) == 3)
check("one block used per layer", len(mgr._sequences[sid].page_table[0]) == 1)

# Append 2 more tokens (crosses block boundary: 3+2=5, block_size=4)
k2, v2 = rand_kv(2)
append_all_layers(mgr, sid, k2, v2)
check("length after append(2)", mgr.get_kv_length(sid) == 5)
check("two blocks used", len(mgr._sequences[sid].page_table[0]) == 2)

# Free
mgr.free_sequence(sid)
check("freed successfully", sid not in mgr._sequences)
check("all blocks returned", mgr.allocator.num_used == 0)


# ============================================================================
# TEST 3: build_packed_kv correctness
# ============================================================================
print("\n=== Test 3: build_packed_kv ===")

mgr = make_mgr()
sid = mgr.allocate_sequence()

# Append 6 tokens across two blocks (block_size=4)
k, v = rand_kv(6)
append_all_layers(mgr, sid, k, v)

# Check layer 0
packed_k, packed_v, cu, max_len = mgr.build_packed_kv([sid], layer_idx=0)
check("packed shape", packed_k.shape == (6, 4, 8), f"got {packed_k.shape}")
check("cu_seqlens", cu.tolist() == [0, 6], f"got {cu.tolist()}")
check("max_seqlen", max_len == 6)

# Verify data matches what we wrote
check("k data matches", torch.allclose(packed_k, k, atol=1e-6))
check("v data matches", torch.allclose(packed_v, v, atol=1e-6))

mgr.free_sequence(sid)


# ============================================================================
# TEST 4: Multi-sequence batched packing
# ============================================================================
print("\n=== Test 4: Multi-sequence packing ===")

mgr = make_mgr()
sid0 = mgr.allocate_sequence()
sid1 = mgr.allocate_sequence()

k0, v0 = rand_kv(3)
k1, v1 = rand_kv(7)
append_all_layers(mgr, sid0, k0, v0)
append_all_layers(mgr, sid1, k1, v1)

packed_k, packed_v, cu, max_len = mgr.build_packed_kv([sid0, sid1], layer_idx=0)
check("total tokens", packed_k.shape[0] == 10, f"got {packed_k.shape[0]}")
check("cu_seqlens", cu.tolist() == [0, 3, 10], f"got {cu.tolist()}")
check("max_seqlen", max_len == 7)

# Verify per-sequence data
check("seq0 k data", torch.allclose(packed_k[:3], k0, atol=1e-6))
check("seq1 k data", torch.allclose(packed_k[3:10], k1, atol=1e-6))

mgr.free_sequence(sid0)
mgr.free_sequence(sid1)


# ============================================================================
# TEST 5: Fork (CoW sharing)
# ============================================================================
print("\n=== Test 5: Fork / CoW ===")

mgr = make_mgr()
parent = mgr.allocate_sequence()

k_parent, v_parent = rand_kv(4)  # exactly one full block
append_all_layers(mgr, parent, k_parent, v_parent)

blocks_before_fork = mgr.allocator.num_used
child = mgr.fork_sequence(parent)

check("child gets same length", mgr.get_kv_length(child) == 4)
check("no new blocks allocated (sharing)", mgr.allocator.num_used == blocks_before_fork,
      f"used: {mgr.allocator.num_used}, expected: {blocks_before_fork}")

# Verify child reads same data
pk_parent, pv_parent, _, _ = mgr.build_packed_kv([parent], layer_idx=0)
pk_child, pv_child, _, _ = mgr.build_packed_kv([child], layer_idx=0)
check("child reads same k", torch.allclose(pk_parent, pk_child, atol=1e-6))
check("child reads same v", torch.allclose(pv_parent, pv_child, atol=1e-6))

# Append to child — should trigger CoW on the tail block
k_new, v_new = rand_kv(1)
append_all_layers(mgr, child, k_new, v_new)
check("child length grows", mgr.get_kv_length(child) == 5)
check("parent length unchanged", mgr.get_kv_length(parent) == 4)

# After CoW, new blocks should have been allocated
check("new block for child's CoW", mgr.allocator.num_used > blocks_before_fork)

# Parent data should be untouched
pk_parent2, _, _, _ = mgr.build_packed_kv([parent], layer_idx=0)
check("parent data unchanged after child append",
      torch.allclose(pk_parent, pk_parent2, atol=1e-6))

# Child data should include parent prefix + new token
pk_child2, pv_child2, _, _ = mgr.build_packed_kv([child], layer_idx=0)
check("child has 5 tokens", pk_child2.shape[0] == 5)
check("child prefix matches parent", torch.allclose(pk_child2[:4], k_parent, atol=1e-6))
check("child new token matches", torch.allclose(pk_child2[4:5], k_new, atol=1e-6))

# Free parent — shared blocks should just get refcount decremented
mgr.free_sequence(parent)
check("parent freed", parent not in mgr._sequences)

# Child should still be valid and readable
pk_child3, _, _, _ = mgr.build_packed_kv([child], layer_idx=0)
check("child still valid after parent freed",
      torch.allclose(pk_child2, pk_child3, atol=1e-6))

mgr.free_sequence(child)
check("all blocks freed", mgr.allocator.num_used == 0)


# ============================================================================
# TEST 6: Multi-fork (simulate branching factor of 3)
# ============================================================================
print("\n=== Test 6: Multi-fork branching ===")

mgr = make_mgr()
parent = mgr.allocate_sequence()
k_shared, v_shared = rand_kv(5)
append_all_layers(mgr, parent, k_shared, v_shared)

children = [mgr.fork_sequence(parent) for _ in range(3)]

# Parent is "deactivated" in UQ — free it
mgr.free_sequence(parent)

# Each child appends a different token
child_tokens = []
for c in children:
    k_c, v_c = rand_kv(1)
    child_tokens.append((k_c, v_c))
    append_all_layers(mgr, c, k_c, v_c)

# Check each child has correct data
for i, c in enumerate(children):
    pk, pv, cu, ml = mgr.build_packed_kv([c], layer_idx=0)
    check(f"child {i} length", pk.shape[0] == 6)
    check(f"child {i} prefix", torch.allclose(pk[:5], k_shared, atol=1e-6))
    check(f"child {i} unique token", torch.allclose(pk[5:6], child_tokens[i][0], atol=1e-6))

# Batched pack of all children
pk_all, pv_all, cu_all, ml_all = mgr.build_packed_kv(children, layer_idx=0)
check("batched total", pk_all.shape[0] == 18)  # 3 * 6
check("batched cu_seqlens", cu_all.tolist() == [0, 6, 12, 18])

for c in children:
    mgr.free_sequence(c)
check("all blocks freed after multi-fork", mgr.allocator.num_used == 0)


# ============================================================================
# TEST 7: Sliding window trim
# ============================================================================
print("\n=== Test 7: Sliding window ===")

mgr = make_mgr(block_size=4)
sid = mgr.allocate_sequence()

# Append 10 tokens (3 blocks: [4] [4] [2])
k, v = rand_kv(10)
append_all_layers(mgr, sid, k, v)

check("pre-trim length", mgr.get_kv_length(sid) == 10)
check("pre-trim blocks", len(mgr._sequences[sid].page_table[0]) == 3)

# Trim to window of 6 (evict first 4 tokens = 1 full block)
mgr.trim_to_window(sid, window_size=6)
check("post-trim length", mgr.get_kv_length(sid) == 6)

pk, pv, cu, ml = mgr.build_packed_kv([sid], layer_idx=0)
check("trimmed data shape", pk.shape[0] == 6)
# The remaining tokens should be k[4:10]
check("trimmed data matches tail", torch.allclose(pk, k[4:10], atol=1e-6))

mgr.free_sequence(sid)


# ============================================================================
# TEST 8: OOM detection
# ============================================================================
print("\n=== Test 8: OOM detection ===")

mgr = make_mgr(max_blocks=2, block_size=4)
sid = mgr.allocate_sequence()

# 4 tokens = 1 block per layer = 2 blocks total
k, v = rand_kv(4)
append_all_layers(mgr, sid, k, v)
check("2 blocks used", mgr.allocator.num_used == 2)

# 1 more token requires a new block — but we only have 2 total
try:
    k2, v2 = rand_kv(1)
    append_all_layers(mgr, sid, k2, v2)
    check("OOM raised", False, "expected RuntimeError")
except RuntimeError as e:
    check("OOM raised", "out of blocks" in str(e).lower())

mgr.free_sequence(sid)


# ============================================================================
# TEST 9: Empty sequence edge case
# ============================================================================
print("\n=== Test 9: Empty sequence ===")

mgr = make_mgr()
sid = mgr.allocate_sequence()

pk, pv, cu, ml = mgr.build_packed_kv([sid], layer_idx=0)
check("empty packed k shape", pk.shape == (0, 4, 8))
check("empty cu_seqlens", cu.tolist() == [0, 0])
check("empty max_seqlen", ml == 0)

mgr.free_sequence(sid)


# ============================================================================
# TEST 10: Consistency across layers
# ============================================================================
print("\n=== Test 10: Cross-layer consistency ===")

mgr = make_mgr(num_layers=3)
sid = mgr.allocate_sequence()

k, v = rand_kv(7)
append_all_layers(mgr, sid, k, v)

for li in range(3):
    pk, pv, cu, ml = mgr.build_packed_kv([sid], layer_idx=li)
    check(f"layer {li} shape", pk.shape[0] == 7)
    check(f"layer {li} data matches", torch.allclose(pk, k, atol=1e-6))

mgr.free_sequence(sid)


# ============================================================================
# TEST 11: Fork then fork (nested branching)
# ============================================================================
print("\n=== Test 11: Nested forking ===")

mgr = make_mgr()
root = mgr.allocate_sequence()
k0, v0 = rand_kv(3)
append_all_layers(mgr, root, k0, v0)

child_a = mgr.fork_sequence(root)
mgr.free_sequence(root)

# Extend child_a
k1, v1 = rand_kv(2)
append_all_layers(mgr, child_a, k1, v1)

# Fork child_a into grandchildren
gc1 = mgr.fork_sequence(child_a)
gc2 = mgr.fork_sequence(child_a)
mgr.free_sequence(child_a)

# Extend grandchildren differently
k_gc1, v_gc1 = rand_kv(1)
k_gc2, v_gc2 = rand_kv(1)
append_all_layers(mgr, gc1, k_gc1, v_gc1)
append_all_layers(mgr, gc2, k_gc2, v_gc2)

pk1, _, _, _ = mgr.build_packed_kv([gc1], layer_idx=0)
pk2, _, _, _ = mgr.build_packed_kv([gc2], layer_idx=0)

expected_prefix = torch.cat([k0, k1], dim=0)
check("gc1 has 6 tokens", pk1.shape[0] == 6)
check("gc2 has 6 tokens", pk2.shape[0] == 6)
check("gc1 prefix matches", torch.allclose(pk1[:5], expected_prefix, atol=1e-6))
check("gc2 prefix matches", torch.allclose(pk2[:5], expected_prefix, atol=1e-6))
check("gc1 unique tail", torch.allclose(pk1[5:6], k_gc1, atol=1e-6))
check("gc2 unique tail", torch.allclose(pk2[5:6], k_gc2, atol=1e-6))

mgr.free_sequence(gc1)
mgr.free_sequence(gc2)
check("all blocks freed after nested fork", mgr.allocator.num_used == 0)


# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*60}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'='*60}")

sys.exit(0 if failed == 0 else 1)