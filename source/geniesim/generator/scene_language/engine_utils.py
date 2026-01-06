from geniesim.generator.scene_language.type_utils import Shape
from geniesim.generator.scene_language.engine.constants import ENGINE_MODE


__all__ = ["primitive_call"]

if ENGINE_MODE == "exposed":
    from geniesim.generator.scene_language._engine_utils_exposed import (
        primitive_call as _primitive_call,
    )
else:
    raise NotImplementedError(ENGINE_MODE)


def primitive_call(name, *args, **kwargs) -> Shape:
    # inner_primitive_call may be updated in `impl_helper.make_new_library`
    return inner_primitive_call(name, *args, **kwargs)


def inner_primitive_call(name, *args, **kwargs) -> Shape:
    kwargs = {k: v for k, v in kwargs.items() if k != "prompt_kwargs_29fc3136"}
    return _primitive_call(name, *args, **kwargs)
