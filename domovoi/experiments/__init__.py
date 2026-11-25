from importlib import import_module

__all__ = [
    "allocate_experiment_id", "create_experiment", "list_experiments",
    "load_metadata", "set_status", "update_metadata", "generate_report",
    "read_metrics", "read_sessions", "start_session",
]


def __getattr__(name):
    # Launch preflight is stdlib-only. Load registry/report dependencies on demand.
    if name not in __all__:
        raise AttributeError(name)
    if name in ("read_metrics", "read_sessions", "start_session"):
        module = ".metrics_log"
    elif name == "generate_report":
        module = ".report"
    else:
        module = ".registry"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
