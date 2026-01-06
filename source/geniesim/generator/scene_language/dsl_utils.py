from __future__ import annotations
from contextlib import contextmanager
from functools import wraps
import numpy as np
import uuid
from typing import Callable, Optional
from geniesim.generator.scene_language.type_utils import Shape, Box, ShapeSampler
from geniesim.generator.scene_language.shape_utils import concat_shapes
import inspect
import random


# __all__ = ['register', 'library_call', 'register_animation']
__all__ = ["register", "library_call"]

# We assume there is only one of these
animation_func = None
library = {}

TRACK_HISTORY = False
LOCK = False
RR = Callable[["RR"], Callable[[int], Shape]]


def mkrec(g: RR) -> Callable[[int], Shape]:
    """
    A recursion helper to avoid the need for explicit recursion in the DSL.

    Returns:
         A function accepting recursion depth to return a shape.
    """
    return g(g)


def loop(n: int, fn: Callable[[int], Shape]) -> Shape:
    """
    Simple loop executing a function `n` times and concatenating the results.

    Args:
        n (int): Number of iterations.
        fn (Callable[[int], Shape]): Function that takes the current iteration index returns a shape.

    Returns:
        Concatenated shapes from each iteration.
    """

    return concat_shapes(*[fn(i) for i in range(n)])


def if_else(c: bool, true_fn: Callable[[], Shape], false_fn: Callable[[], Shape]) -> Shape:
    """
    Executes one of two functions based on a boolean condition, emulating an 'if-else' statement.

    Args:
        c: bool - Condition determining which function to execute.
        true_fn: Callable - Function to execute if the condition is True.
        false_fn: Callable - Function to execute if the condition is False.

    Returns:
        Shape returned by the executed function.
    """
    return true_fn() if c else false_fn()


def get_caller_name(self: Optional[str]) -> str:
    for depth in range(1, len(inspect.stack())):
        caller = inspect.stack()[depth]
        if caller.function == "library_call":
            caller_args = inspect.getargvalues(caller.frame)
            caller = caller_args.locals["func_name"]
            if self is not None and caller == self:
                # this case should only be hit once
                continue
            else:
                break
        else:
            caller = caller.function
            if caller not in library:
                continue
            else:
                break
    return caller


def register(docstring: Optional[str] = None):
    """
    Registers a function whose name must be unique. You can pass in a docstring (optional).

    Every function you register MUST be invoked via `library_call`, and cannot be invoked directly via the function name.
    """

    def decorator(func: ShapeSampler) -> ShapeSampler:
        if LOCK is True:
            print(f"Skipping registration of {func.__name__}")
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            # caller = get_caller_name(func.__name__)
            # print(f'{caller=} calls {func.__name__}')
            ret = func(*args, **kwargs)  # FIXME should use the function in the library
            if LOCK is False:  # and the call is successful
                library[func.__name__]["last_call"] = (args, kwargs)

            call_id = uuid.uuid4()
            if ret is None:
                # something is wrong
                print(f"[ERROR] {func.__name__} returned None")
                return ret
            for elem in ret:
                elem["info"]["stack"].append((func.__name__, call_id))

            if TRACK_HISTORY:
                library[func.__name__]["hist_calls"].append((args, kwargs, get_caller_name(func.__name__)))
            # print(f'[INFO] calling {func.__name__}', library[func.__name__]['hist_calls'][-1])
            if len(library[func.__name__]["hist_calls"]) > 1000:
                print(f"[WARNING] {func.__name__} has more than 1000 calls")
            #     library[func.__name__]['hist_calls'].pop(0)
            return ret

        if func.__name__ in library:
            # not sure if this case is possible
            print(f"[ERROR] shape {func.__name__} already exists in {library.keys()}")
            # probably something went wrong
            # return wrapper  # TODO not sure..

        library[func.__name__] = {
            "__target__": wrapper,
            "docstring": docstring if docstring is not None else func.__name__,
            "check": Box((0, 0, 0), 1),  # hack
            "last_call": None,
            "hist_calls": [],
        }

        return wrapper

    return decorator


# def register_frame():
#     """
#     Decorates a function with return type `Shape`. The function returns the final scene.
#     """
#     return register()


# def register_frames():
#     """
#     Decorates a function with return type `Generator[Shape, None, None]`.
#     The function yields frames of an animated scene.
#     """
#     return register_animation()


def register_animation(docstring: Optional[str] = None):
    """
    Registers an animation function which is stored in the global `animation_func`. You can pass an optional docstring.

    If you register a function, there a couple of rules:
        - That function should never be called anywhere else in the program. This function gets used later by the rendering engine.
        - This function needs a return type of `Generator[Shape, None, None]`.
    """

    def decorator(func: Callable) -> Callable:
        global animation_func

        if animation_func is not None:
            print(f"[ERROR] An animation function is already registered.")
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if result is None:
                print(f"[ERROR] {func.__name__} returned None")

            return result

        animation_func = wrapper
        animation_func.__doc__ = docstring if docstring else f"{func.__name__} does not have a docstring."
        return wrapper

    return decorator


def animation_library_call(new_library=None) -> list[Shape]:
    if animation_func is None:
        print("[INFO] No animation function is registered.")
        return None

    if new_library is None:
        return list(animation_func())

    orig_library = library.copy()
    library.clear()
    library.update(new_library)
    ret = list(animation_func())
    library.clear()
    library.update(orig_library)
    return ret


FAKE_CALL = False
_children = set()  # user is responsible for clearing


def library_call(func_name: str, **kwargs) -> Shape:
    """
    Call a function from the library and return its outputs. You are responsible for registering the function with `register`.

    Args:
        func_name (str): Function name.
        **kwargs: Keyword arguments passed to the function.
    """
    if FAKE_CALL:
        _children.add(func_name)
        return []
    if func_name not in library:
        for alt_func_name in library.keys():
            if library[alt_func_name]["docstring"] == func_name:
                # print(f'WARNING: {func_name=} not found in library but found an alternative: {alt_func_name=}')
                # with set_seed(0):
                return library[alt_func_name]["__target__"](**kwargs)

        for alt_func_name in library.keys():
            if library[alt_func_name]["docstring"].split(";")[0] == func_name:
                # with set_seed(0):
                return library[alt_func_name]["__target__"](**kwargs)
        print(f"WARNING: {func_name=} not found in library")

        return []
    # with set_seed(0):
    return library[func_name]["__target__"](**kwargs)


def clear_history():
    for name in library.keys():
        library[name]["hist_calls"].clear()


@contextmanager
def set_track_history_enabled(mode: bool):
    global TRACK_HISTORY
    orig_track_history = TRACK_HISTORY
    TRACK_HISTORY = mode
    try:
        yield TRACK_HISTORY
    finally:
        TRACK_HISTORY = orig_track_history


@contextmanager
def set_lock_enabled(mode: bool):
    global LOCK
    orig_lock = LOCK
    LOCK = mode
    try:
        yield LOCK
    finally:
        LOCK = orig_lock


@contextmanager
def set_fake_call_enabled(mode: bool):
    global FAKE_CALL
    orig_fake_call = FAKE_CALL
    FAKE_CALL = mode
    try:
        yield _children
    finally:
        FAKE_CALL = orig_fake_call


@contextmanager
def set_seed(seed: int):
    state_np = np.random.get_state()
    np.random.seed(seed)
    state_random = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state_np)
        random.setstate(state_random)
