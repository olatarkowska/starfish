"""
Because multiprocessing.Pool demands that any shared-memory constructs be passed in as initializer
arguments, and not as arguments to multiprocessing.Pool.map or multiprocessing.Pool.apply, we need a
stub initializer that accepts the shared-memory construct and store it in a global for retrieval.
This ugly bit of logic is best kept away from other code.
"""


_global = None


def initializer(payload) -> None:
    global _payload
    _payload = payload


def get_payload():
    return _payload
