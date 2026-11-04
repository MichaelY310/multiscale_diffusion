"""Stable DCP envelope for the native loader's dynamically sized buckets.

DCP's default flattening plans from a fresh empty bucket structure and cannot
load populated tensor/list keys. Serialize only the data state in one bytes
entry; preserve tensor dtype/shape/content and verify SHA256 before safe loading.
Model/optimizer tensors remain ordinary distributed DCP states.
"""
import hashlib
import io

import torch


class NativeDCPDataLoader:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        iter(self.loader)
        return self

    def __next__(self):
        return next(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def state_dict(self):
        buffer = io.BytesIO()
        torch.save(self.loader.state_dict(), buffer)
        payload = buffer.getvalue()
        return {"schema": 1, "sha256": hashlib.sha256(payload).hexdigest(), "payload": payload}

    def load_state_dict(self, state):
        if state.get("schema") != 1 or not isinstance(state.get("payload"), bytes):
            raise ValueError("Invalid native DCP data envelope")
        if hashlib.sha256(state["payload"]).hexdigest() != state["sha256"]:
            raise ValueError("Native DCP data payload hash mismatch")
        data = torch.load(io.BytesIO(state["payload"]), map_location="cpu", weights_only=True)
        self.loader.load_state_dict(data)
