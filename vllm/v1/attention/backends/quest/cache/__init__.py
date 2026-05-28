# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quest backend's GPU/CPU tiered KV cache layer.

All members are backend-private. The orchestrator is `TierManager`; the
others are pure data structures the manager owns.
"""
