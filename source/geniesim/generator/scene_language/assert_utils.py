from geniesim.generator.scene_language.calc_utils import _attach, _align
from geniesim.generator.scene_language.type_utils import P, Shape
from typing import Union, Optional, Callable
from geniesim.generator.scene_language._shape_utils import compute_bbox, compute_bboxes
from collections import Counter
from contextlib import contextmanager
from geniesim.generator.scene_language.dsl_utils import library_call, get_caller_name
import numpy as np
import threading
import time
import uuid


__all__ = [
    "assert_connect",
    "assert_disconnect",
    "assert_attach",
    "assert_align_with_min",
    "assert_align_with_center",
    "assert_components",
    "assert_bound",
    "test_library_call",
]

shape_tested: Shape = None
child_shapes: dict[str, list[Shape]] = None


@contextmanager
def test_library_call(func_name: str, **kwargs) -> Shape:
    """
    A context manager; test a function from the library and yield its outputs.

    Args:
        func_name (str): Function name.
        **kwargs: Keyword arguments passed to the function.
    """
    import mi_helper  # FIXME hack should fix later; this is to call `primitive_call.implement`

    global shape_tested, child_shapes
    shape = library_call(func_name, **kwargs)
    shape_tested = shape
    child_shapes = retrieve_child_shapes(shape)
    timeout = False

    def timeout_checker():
        nonlocal timeout
        time.sleep(2)
        if not timeout:
            print("time is out!")

    timeout_thread = threading.Thread(target=timeout_checker)
    timeout_thread.start()

    try:
        yield shape
    finally:
        timeout = True
        shape_tested = None
        child_shapes = None
        timeout_thread.join()


def retrieve_child_shapes(shape: Shape) -> dict[str, list[Shape]]:
    ret = {}
    for elem in shape:
        if len(elem["info"]["stack"]) == 1:
            # not sure, hack
            name, call_id = "primitive", uuid.uuid4()
        else:
            name, call_id = elem["info"]["stack"][-2]
        # print(elem['info']['stack'])
        if name not in ret:
            ret[name] = {}
        if call_id not in ret[name]:
            ret[name][call_id] = []
        ret[name][call_id].append(elem)
    for name in ret:
        ret[name] = list(ret[name].values())

    # print('retrieved shapes', {k: len(v) for k, v in ret.items()})
    return ret


def assert_connect(shapes: list[str]):
    """
    Assert that all shapes have at least one intersecting 3D point.
    This is a stronger constraint than pairwise intersection.
    """
    # TODO
    pass


def assert_disconnect(shapes: list[str]):
    """
    Assert that any pair of shapes have no intersecting 3D points.
    """
    # TODO
    pass


def assert_bound(direction: P, bmin: Union[float, str, None], bmax: Union[float, str, None], shape: str):
    """
    Asserts that the shape's extent along the direction is bounded by 'bmin' and 'bmax'.
    If 'bmin' ('bmax') is a string, the bound is computed as the min (max) projection of ANY shape instance with this name.
    This function makes assertion for ALL shape instances with name `shape`.
    """
    atol = 1e-3
    all_shapes = child_shapes

    find_minmax = lambda b: (b.min @ direction, b.max @ direction)

    if isinstance(bmin, list):
        bmin = min(find_minmax(compute_bbox(bmin)))
    elif isinstance(bmin, str):
        if bmin not in all_shapes:
            print(f"[FAILED] shape not found: {bmin=}")
            return
        bmin_boxes = [compute_bbox(c) for c in all_shapes[bmin]]
        bmin = min([min(find_minmax(b)) for b in bmin_boxes])
    elif bmin is None:
        bmin = -np.inf

    if isinstance(bmax, list):
        bmax = max(find_minmax(compute_bbox(bmax)))
    elif isinstance(bmax, str):
        if bmax not in all_shapes:
            print(f"[FAILED] shape not found: {bmax=}")
            return
        bmax_boxes = [compute_bbox(c) for c in all_shapes[bmax]]
        bmax = max([max(find_minmax(b)) for b in bmax_boxes])
    elif bmax is None:
        bmax = np.inf

    # ALL shapes with the given name must be within bounds
    success = True
    for targ in all_shapes[shape]:
        actual_bounds = find_minmax(compute_bbox(targ))
        if bmin - atol <= min(actual_bounds) and max(actual_bounds) <= bmax + atol:
            pass
        else:
            caller = get_caller_name(None)
            print(
                f"[FAILED] shape not bounded: {shape=} {caller=} {direction=}; expected {bmin} <= {actual_bounds} <= {bmax}"
            )
            success = False
    if success:
        print(f"[PASSED] shape instance(s) {shape=} are bounded within [{bmin}, {bmax}] along {direction=}")


def assert_attach(direction: P, shapes: list[str]):
    """
    Asserts that each shape is attached sequentially to the next along the given direction.
    """
    success = _assert_reduce(
        lambda s1, s2: not _attach(direction, [s1, s2], atol=1e-1)[1],
        shapes,
        prev_target=None,
        all_shapes=child_shapes,
    )
    if not success:
        caller = get_caller_name(None)
        print(f"[FAILED] shape not attached: {caller=} {direction=} {shapes=}")
        return
    print(f"[PASSED] shape attached: {direction=} {shapes=}")


def _assert_reduce_orderless(
    fn: Callable[[list[Shape]], bool],
    shapes: list[str],
    all_shapes: dict[str, list[Shape]],
) -> bool:
    # TODO maybe shuffle and run the test multiple times
    shapes_to_test = []
    for k, ct in Counter(shapes).items():
        if k not in all_shapes:
            print(f"[FAILED] shape not found: {k=}")
            return False
        if len(all_shapes[k]) < ct:
            print(f"[FAILED] not enough shapes: {k=}")
            return False
        shapes_to_test.extend(all_shapes[k][:ct])
    return fn(shapes_to_test)


def _assert_reduce(
    fn: Callable[[Shape, Shape], bool],
    shapes: list[str],
    prev_target: Optional[Shape],
    all_shapes: dict[str, list[Shape]],
) -> bool:
    # if sum(len(v) for v in all_shapes.values()) > 10:
    #     print('[FAILED] Too many shapes to test')
    #     return False
    if len(shapes) == 0:
        return True
    success = False
    first_shape = shapes[0]
    rest_shapes = shapes[1:]
    if first_shape not in all_shapes:
        print(f"[FAILED] shape not found: {first_shape=}")
        return False
    for ind, target in enumerate(all_shapes[first_shape]):
        if prev_target is None:
            cur_success = True
        else:
            cur_success = fn(prev_target, target)
        if cur_success and _assert_reduce(
            fn=fn,
            shapes=rest_shapes,
            prev_target=target,
            all_shapes={
                first_shape: all_shapes[first_shape][:ind] + all_shapes[first_shape][ind + 1 :],
                **{k: v for k, v in all_shapes.items() if k != first_shape},
            },
        ):
            success = True  # if ANY path succeeds, the test passes
            break
        else:
            continue
    if not success:
        # if prev_target is None:
        # import ipdb; ipdb.set_trace()
        return False
    return True


def assert_align_with_min(normal: P, shapes: list[str]):
    """
    Asserts that the 'minimum' points of the shapes, when projected to the direction of the normal vector, coincide at the same location.
    Note: Reversing the normal swaps 'minimum' with 'maximum'.
    Runs test for ANY combination of shape instances with the given name `shapes`.
    """
    success = _assert_reduce_orderless(
        fn=lambda shapes: not _align("min", normal, shapes, atol=1e-2)[1],
        shapes=shapes,
        all_shapes=child_shapes,
    )
    # success = _assert_reduce(fn=lambda s1, s2: not _align('min', normal, [s1, s2], atol=1e-2)[1],
    #                          shapes=shapes, prev_target=None, all_shapes=retrieve_shapes())
    if not success:
        caller = get_caller_name(None)
        print(f"[FAILED] shape not aligned with min: {caller=} {normal=} {shapes=}")
        return
    print(f"[PASSED] shapes {shapes=} aligned with min: {normal=} {shapes=}")


def assert_align_with_center(normal: P, shapes: list[str]):
    """
    Asserts that the center points of the shapes, when projected to the direction of the normal vector, coincide at the same location.
    Runs test for ANY combination of shape instances with the given name `shapes`.
    """
    success = _assert_reduce_orderless(
        fn=lambda shapes: not _align("center", normal, shapes, atol=1e-2)[1],
        shapes=shapes,
        all_shapes=child_shapes,
    )
    # success = _assert_reduce(fn=lambda s1, s2: not _align('center', normal, [s1, s2], atol=1e-2)[1],
    #                          shapes=shapes, prev_target=None, all_shapes=retrieve_shapes())
    if not success:
        caller = get_caller_name(None)
        print(f"[FAILED] shape not aligned with center: {caller=} {normal=} {shapes=}")
        return
    print(f"[PASSED] shapes {shapes=} aligned with center: {normal=} {shapes=}")


def assert_components(shapes: list[str]):
    """
    Asserts that the scene contains exactly the specified shape instances.
    """
    real_counts = [(k, len(v)) for k, v in sorted(child_shapes.items()) if len(v) > 0]
    targ_counts = list(sorted(Counter(shapes).items()))
    if real_counts != targ_counts:
        print(f"[FAILED] assert_components expected {targ_counts}, got {real_counts}")
        return
    print(f"[PASSED] assert_components {real_counts}")


def assert_freeform(constr: str, shapes: list[str]):
    """
    Assert that the given free-form constraint holds for the given shapes.
    """
    pass


def eq(s1: Shape, s2: Shape) -> bool:
    atol = 1e-2
    if len(s1) != len(s2):
        print(f"Expected: {len(s1)}, got: {len(s2)}")
        return False
    for ss1, ss2 in zip(s1, s2):
        if not np.allclose(ss1["to_world"], ss2["to_world"], atol=atol):
            print(f"Expected: {ss1}, got: {ss2}")
            print("[FAILED] in eq", ss1["info"], ss2["info"])
            print(np.linalg.norm(ss1["to_world"] - ss2["to_world"]))
            return False
    return True


# def retrieve_shapes() -> dict[str, list[Shape]]:
#     caller = get_caller_name(None)
#     # print(f'checking for {caller=}')
#
#     from geniesimgen.dsl_utils import library
#     ret: dict[str, list[Shape]] = {}
#     for name, func in library.items():
#         # print(name, [c for _, _, _, c, _ in func['hist_calls']])
#         ret[name] = [s for _, _, s, c, _ in func['hist_calls'] if c == caller]
#     return ret
