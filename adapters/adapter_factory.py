from adapters.llama_adapter import LlamaAdapter 
from adapters.qwen3_adapter import Qwen3Adapter
from adapters.model_adapter import ModelAdapter


# Map from HuggingFace config.model_type → adapter class.
_ADAPTER_REGISTRY: dict[str, type] = {
    "qwen3": Qwen3Adapter,
    "llama": LlamaAdapter,
}


def get_adapter(model) -> ModelAdapter:
    """
    Instantiate and return the correct ModelAdapter for *model*.

    Detection order:
      1. model.config.model_type matched against the registry.
      2. Class-name heuristics as a fallback for unregistered types.

    Raises:
        NotImplementedError: if no adapter is found for the model family.
    """
    model_type = getattr(model.config, "model_type", "").lower()

    if model_type in _ADAPTER_REGISTRY:
        return _ADAPTER_REGISTRY[model_type](model)

    # Fallback: heuristic on the class name (handles custom subclasses)
    cls_name = type(model).__name__.lower()
    if "qwen3" in cls_name or "qwen" in cls_name:
        return Qwen3Adapter(model)
    if "llama" in cls_name:
        return LlamaAdapter(model)

    raise NotImplementedError(
        f"No ModelAdapter registered for model_type={model_type!r} "
        f"(class={type(model).__name__!r}).  "
        f"Registered types: {sorted(_ADAPTER_REGISTRY.keys())}.  "
        f"Add a new ModelAdapter subclass and register it in _ADAPTER_REGISTRY."
    )


def register_adapter(model_type: str, adapter_cls: type) -> None:
    """
    Register a custom adapter for a new model family at runtime.

    Example:
        from model_adapter import register_adapter, ModelAdapter
        register_adapter("my_model_type", MyModelAdapter)
    """
    _ADAPTER_REGISTRY[model_type.lower()] = adapter_cls