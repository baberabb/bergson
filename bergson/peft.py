from peft import PeftModel, get_peft_model_state_dict
from transformers import PreTrainedModel


def detect_peft_modules(model: PeftModel) -> set[str]:
    # Extract target modules
    target_modules = set()
    peft_state_dict = get_peft_model_state_dict(model)

    for adapter_name in model.peft_config:
        for name in peft_state_dict:
            prefix = name.removesuffix(".weight")
            processed_name = f"{prefix}.{adapter_name}".removeprefix("base_model.")
            try:
                model.get_submodule(processed_name)
                target_modules.add(processed_name)
            except AttributeError as e:
                raise RuntimeError(
                    f"Adapter parameter '{processed_name}' not found in the model."
                ) from e

    return target_modules


def set_peft_enabled(model: PreTrainedModel, enabled: bool):
    """Bizarrely, the standard enable_adapters and disable_adapters aren't working"""
    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils import ModulesToSaveWrapper

    futile = True

    for _, module in model.named_modules():
        if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
            futile = False
            module.enable_adapters(enabled=enabled)

    if futile:
        raise RuntimeError("PEFT model incorrectly initialized")
