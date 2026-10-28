# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared USD ``render_layer.usda`` authoring helpers.

An assembler authors a ``render_layer.usda`` for the OVRTX render node, and
the *camera-level* authoring is the same whichever assembler does it. The Kit
path is :mod:`genie_sim_engine_isaacsim.assemble._scene_impl`, which also
supports "mirror mode" (copying intrinsics / xform / materials out of a
prebaked ``robot.usda``); that part stays in that module. A Kit-free
assembler synthesizes cameras from the scene YAML instead.

Before this module a second assembler carried a byte-for-byte duplicate of the
pose / synthesized-intrinsic / fisheye / RenderProduct / FreeCam authoring, so
a new camera USD attribute (lens model, projection, …) had to be added in two
places or it would work under one physics engine and silently not the other
(the ``model: opencvFisheye`` regression was exactly this). Keeping the shared
authoring here makes every assembler author identical output from one
source.

``pxr`` is imported lazily inside each function on purpose: importing this
module must not require a running Kit / USD runtime, matching the convention in
``_scene_impl.py`` where every pxr-touching helper gates its own import.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Mapping, Optional

# Isaac Sim's default aperture for stock cameras. Pinning ``horizontalAperture``
# to this makes the rendered fx match the YAML fx exactly (see
# :func:`synthesize_camera_intrinsics`).
DEFAULT_HORIZONTAL_APERTURE = 20.955

# Lens LUT tables are baked at this fraction of the render resolution. Measured,
# not guessed: against full-resolution tables a 0.25x table shifts the rendered
# frame by mean 0.134 / 0.139 luma (1920x1536 head lenses), which is *below* the
# path tracer's own frame-to-frame noise floor of 0.228 measured between two
# renders of the same lens; at the image-circle rim, where resampling error would
# show first, the difference is 0.009. It also takes the three head lenses from
# 144 MB of tables to 9.7 MB.
DEFAULT_LENS_LUT_SCALE = 0.25

# Subdirectory of the render_layer.usda output dir that holds baked lens tables.
LENS_LUT_SUBDIR = "lens"


def _kb3_from_opencv_fisheye(fx: float, k: "list[float]", max_fov_deg: float) -> "list[float]":
    """Refit an OpenCV 4-coefficient fisheye onto NVIDIA's 3-coefficient KannalaBrandtK3.

    **Unused — kept only as documentation of a dead end.** ``kannalaBrandtK3``
    was attractive because it is the one Kannala-Brandt schema that exposes a
    ``maxFov`` knob, which would have let these ~221 deg lenses render their
    full field instead of being clipped at the 180 deg hemisphere that
    ``opencvFisheye`` enforces. It does not work: the model is deprecated in
    this ovrtx build and *non-functional*, not merely warned about — a probe
    render comes out a uniform flat field (std 0.93 over the whole frame) with
    no scene content at all. Same for ``ftheta`` (pure black) and
    ``radTanThinPrism``. The only working models are ``opencvFisheye``, which
    is capped at 180 deg, and ``lut``, which has no cap; going past the
    hemisphere means moving to ``lut``, not to this.

    Both models are the same Kannala-Brandt odd-power series in the incident
    angle ``theta``, but they differ in how many terms they carry and where the
    focal length lives::

        OpenCV : r = fx * theta * (1 + k1*t^2 + k2*t^4 + k3*t^6 + k4*t^8)
        NVIDIA : r =      k0*theta +  k1*t^3 +  k2*t^5 +  k3*t^7

    So NVIDIA has no slot for OpenCV's ``k4``. The naive term-by-term mapping
    (``k0=fx``, ``kN=fx*kN_cv``) therefore silently *drops* it, and for these
    ~221 deg lenses that is not a rounding error: it pushes the image-circle
    radius from ~1000 px out to ~1260-1330 px. Instead we least-squares fit the
    four NVIDIA coefficients against the exact OpenCV curve over the used range
    ``theta in [0, max_fov_deg/2]``, which reproduces it to well under a pixel
    RMS (~3 px worst case at the very rim).
    """
    import numpy as np

    k1, k2, k3, k4 = (list(k) + [0.0, 0.0, 0.0, 0.0])[:4]
    theta = np.linspace(0.0, np.radians(max_fov_deg / 2.0), 4001)
    r_true = fx * theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
    basis = np.stack([theta, theta**3, theta**5, theta**7], axis=1)
    coef, *_ = np.linalg.lstsq(basis, r_true, rcond=None)
    return [float(c) for c in coef]


def author_opencv_fisheye(prim, intrinsic: Mapping[str, Any], width: int, height: int) -> None:
    """Apply ``OmniLensDistortionOpenCvFisheyeAPI`` + intrinsics so ovrtx renders fisheye distortion.

    The ovrtx RTX delegate consumes ``omni:lensdistortion:*`` on a Camera prim.
    The API schema is codeless and not registered in the system pxr that runs
    the assemblers, so we author the ``apiSchemas`` list-op directly (the attrs
    come out ``custom``); ovrtx's bundled USD recognises it at render time.
    ``imageSize`` is pinned to the RenderProduct resolution — a mismatch makes
    the fisheye projection crop / letterbox.

    This schema has **no FOV knob** and ovrtx renders it to a 180 deg
    hemisphere only. Pixels past that get no traced ray and simply show
    whatever the scene environment is (bright white in the supermarket scene),
    so the render node paints them black afterwards — see
    ``fisheye_mask_radius_px`` in ``render_node.cpp``, fed by the scene YAML's
    ``mask_fov_deg``. A lens whose real field is wider than 180 deg must go
    through :func:`author_lut_fisheye` instead; see
    :func:`_kb3_from_opencv_fisheye` for why the FOV-capable Kannala-Brandt
    schema is not an option.
    """
    from pxr import Sdf, Gf

    schema = Sdf.TokenListOp()
    schema.prependedItems = ["OmniLensDistortionOpenCvFisheyeAPI"]
    prim.SetMetadata("apiSchemas", schema)

    prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("opencvFisheye")
    pfx = "omni:lensdistortion:opencvFisheye"
    for key in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"):
        prim.CreateAttribute(f"{pfx}:{key}", Sdf.ValueTypeNames.Float).Set(float(intrinsic.get(key, 0.0)))
    prim.CreateAttribute(f"{pfx}:imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(int(width), int(height)))


def author_opencv_pinhole(prim, intrinsic: Mapping[str, Any], width: int, height: int) -> None:
    """Apply ``OmniLensDistortionOpenCvPinholeAPI`` so ovrtx renders pinhole distortion.

    Counterpart of :func:`author_opencv_fisheye` for lenses that calibrate as an
    OpenCV *pinhole* rather than Kannala-Brandt -- the head stereo pair. The
    schema takes OpenCV's **rational** model,

        r_d = r_u (1 + k1 u + k2 u^2 + k3 u^3) / (1 + k4 u + k5 u^2 + k6 u^3)

    with ``u = r_u^2``, plus the tangential pair ``p1``/``p2``. These are the
    FORWARD (add-distortion) coefficients, i.e. OpenCV ``distCoeffs`` order
    ``k1 k2 p1 p2 k3 k4 k5 k6``, which is what a factory intrinsics JSON ships --
    so the yaml numbers go straight through with no inversion.

    All six ``k`` terms are authored. Dropping ``k4``/``k5``/``k6`` does not
    degrade the fit gracefully, it changes the model: for the head stereo lens
    the numerator alone reaches 3.62 at ``u = 1`` where the full ratio is 0.754,
    turning a barrel curve into a pincushion one.

    ``imageSize`` is pinned to the RenderProduct resolution for the same reason
    as the fisheye path -- a mismatch rescales the distortion field.
    """
    from pxr import Sdf, Gf

    schema = Sdf.TokenListOp()
    schema.prependedItems = ["OmniLensDistortionOpenCvPinholeAPI"]
    prim.SetMetadata("apiSchemas", schema)

    prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("opencvPinhole")
    pfx = "omni:lensdistortion:opencvPinhole"
    for key in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"):
        prim.CreateAttribute(f"{pfx}:{key}", Sdf.ValueTypeNames.Float).Set(float(intrinsic.get(key, 0.0)))
    prim.CreateAttribute(f"{pfx}:imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(int(width), int(height)))


# =========================================================================
# ``lut`` lens model — the only way to render past the 180 deg hemisphere
# =========================================================================
#
# ovrtx exposes exactly one non-deprecated lens model with no angular cap:
# ``lut``. That is not inferred from behaviour, it is what the renderer itself
# says — ``strings librtx.hydra.so`` carries
#
#     Camera '%s': Lens distortion model '<x>' is deprecated. Please use
#     'opencvFisheye', 'opencvPinhole', 'lut', or 'omniDirectionalStereo'
#     instead.
#
# for each of ``ftheta`` / ``kannalaBrandtK3`` / ``radTanThinPrism``, and those
# three render a flat field or pure black in this build. ``opencvFisheye`` works
# but stops at the hemisphere. So a >180 deg lens is ``lut`` or nothing.
#
# ``OmniLensDistortionLutAPI`` drives the projection from two float32 EXR tables:
#
#   rayEnterDirectionTexture  RGB32F, indexed by NDC, holds the unit view
#                             direction for that pixel. (0, 0, +1) — a ray
#                             pointing *behind* the camera — is the renderer's
#                             out-of-FOV sentinel.
#   rayExitPositionTexture    RG32F, indexed by the *octahedral encoding* of a
#                             view direction, holds the NDC that direction lands
#                             on. Outside [0, 1] means off-screen.
#
# The tables below are baked from the same OpenCV/Kannala-Brandt coefficients
# ``author_opencv_fisheye`` hands to the built-in model, so inside the 180 deg
# cone the two agree to within the path tracer's noise (measured: mean luma
# 215.98 vs 215.97 over r <= 817 px). The difference is everything outside it.


def _exr_attribute(name: str, type_name: str, payload: bytes) -> bytes:
    return name.encode() + b"\0" + type_name.encode() + b"\0" + struct.pack("<i", len(payload)) + payload


def _write_exr(path: Path, planes: "dict[str, Any]") -> Path:
    """Write float32 ``planes`` (channel name -> 2-D array) as an uncompressed scanline EXR.

    Hand-rolled on purpose. NVIDIA's reference generator uses the ``OpenEXR``
    module, which is not installed in this runtime image, and the obvious
    substitute — ``cv2`` — gates its OpenEXR codec behind an
    ``OPENCV_IO_ENABLE_OPENEXR`` environment variable that has to be set before
    the first ``import cv2`` *anywhere in the process*. A shared library module
    cannot guarantee that, and when it loses the race ``cv2.imwrite`` just
    returns False. Uncompressed scanline EXR is a short, fully specified format
    and these tables are small, so writing it directly removes the dependency
    and the failure mode together.

    EXR stores channels in alphabetical order, hence the sort.
    """
    import numpy as np  # noqa: PLC0415

    names = sorted(planes)
    arrays = [np.ascontiguousarray(planes[n], dtype="<f4") for n in names]
    height, width = arrays[0].shape

    channels = (
        b"".join(
            # pixelType=2 (FLOAT), pLinear=0, 3 reserved bytes, xSampling=1, ySampling=1
            n.encode() + b"\0" + struct.pack("<iBxxxii", 2, 0, 1, 1)
            for n in names
        )
        + b"\0"
    )

    header = b"\x76\x2f\x31\x01" + struct.pack("<I", 2)  # magic 20000630, version 2, no flags
    header += _exr_attribute("channels", "chlist", channels)
    header += _exr_attribute("compression", "compression", b"\x00")  # NO_COMPRESSION
    header += _exr_attribute("dataWindow", "box2i", struct.pack("<iiii", 0, 0, width - 1, height - 1))
    header += _exr_attribute("displayWindow", "box2i", struct.pack("<iiii", 0, 0, width - 1, height - 1))
    header += _exr_attribute("lineOrder", "lineOrder", b"\x00")  # INCREASING_Y
    header += _exr_attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += _exr_attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    header += _exr_attribute("screenWindowWidth", "float", struct.pack("<f", 1.0))
    header += b"\0"

    row_bytes = width * len(names) * 4
    chunk_bytes = 8 + row_bytes  # int32 y + int32 payload size + payload
    base = len(header) + 8 * height
    offsets = b"".join(struct.pack("<Q", base + y * chunk_bytes) for y in range(height))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(offsets)
        for y in range(height):
            fh.write(struct.pack("<ii", y, row_bytes))
            for a in arrays:
                fh.write(a[y].tobytes())
    return path


def _theta_d(theta, k):
    """OpenCV fisheye forward series: ``theta * (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)``."""
    t2 = theta * theta
    return theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))


def _theta_d_prime(theta, k):
    t2 = theta * theta
    return 1.0 + 3 * k[0] * t2 + 5 * k[1] * t2**2 + 7 * k[2] * t2**3 + 9 * k[3] * t2**4


def _theta_d_extended(theta, k, theta_max):
    """``_theta_d`` inside the calibrated domain, C1 linear continuation outside.

    The polynomial only means anything on ``[0, theta_max]``; these coefficients
    alternate sign and bend back down past it. Continuing with a straight line at
    the boundary slope keeps the forward map strictly increasing, so directions
    outside the lens keep mapping monotonically *further* off-screen instead of
    folding back onto valid NDC and smearing the rim.
    """
    import numpy as np  # noqa: PLC0415

    r_edge = float(_theta_d(np.array(theta_max), k))
    slope = float(_theta_d_prime(np.array(theta_max), k))
    return np.where(
        theta <= theta_max,
        _theta_d(np.minimum(theta, theta_max), k),
        r_edge + slope * (theta - theta_max),
    )


def _invert_theta_d(r, k, theta_max, iterations: int = 8):
    """Solve ``_theta_d(theta) = r`` on ``[0, theta_max]``. Returns ``(theta, valid)``."""
    import numpy as np  # noqa: PLC0415

    probe = np.linspace(0.0, theta_max, 4097)
    if np.any(_theta_d_prime(probe, k) <= 0.0):
        raise ValueError(
            "OpenCV fisheye polynomial is not strictly increasing over the calibrated "
            "half-angle, so it has no inverse and cannot be baked into a lens LUT"
        )
    r_max = float(_theta_d(np.array(theta_max), k))
    valid = np.isfinite(r) & (r >= 0.0) & (r <= r_max)

    theta = np.zeros_like(r, dtype=np.float64)
    np.multiply(r, theta_max / r_max, out=theta, where=valid)  # secant seed; _theta_d(0) == 0
    for _ in range(iterations):
        step = np.zeros_like(theta)
        np.divide(_theta_d(theta, k) - r, _theta_d_prime(theta, k), out=step, where=valid)
        theta -= step
    worst = float(np.max(np.abs(_theta_d(theta, k) - r)[valid])) if np.any(valid) else 0.0
    if worst > 1e-10:
        raise RuntimeError(f"lens LUT inverse did not converge (worst residual {worst:.3e})")
    return theta, valid


def _oct_to_unit_vector(x, y):
    """Octahedral (x, y) in [-1, 1] -> unit vector. NVIDIA's decode, kept verbatim.

    This has to match the renderer's encode bit for bit, so it is deliberately
    not "cleaned up".
    """
    import numpy as np  # noqa: PLC0415

    dx, dy = np.meshgrid(x, y)
    dz = 1.0 - np.abs(dx) - np.abs(dy)
    sx = 2 * np.heaviside(dx, 1) - 1
    sy = 2 * np.heaviside(dy, 1) - 1
    tmp = dx
    dx = np.where(dz <= 0, (1 - np.abs(dy)) * sx, dx)
    dy = np.where(dz <= 0, (1 - np.abs(tmp)) * sy, dy)
    n = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
    return dx * n, dy * n, dz * n


def _texel_centers(size: int):
    """Match the renderer's ``(dispatchThreadId + 0.5) / dimensions`` sampling."""
    import numpy as np  # noqa: PLC0415

    return (np.arange(size, dtype=np.float64) + 0.5) / size


def _lut_enter_table(*, tex_w, tex_h, width, height, fx, fy, cx, cy, k, theta_max):
    """NDC -> unit view direction (camera looks -Z). Extrapolated past the lens rim."""
    import numpy as np  # noqa: PLC0415

    sx = (_texel_centers(tex_w) - cx / width) * width / fx
    sy = -(_texel_centers(tex_h) - cy / height) * height / fy
    X, Y = np.meshgrid(sx, sy)
    R = np.hypot(X, Y)
    theta, valid = _invert_theta_d(R, k, theta_max)
    inv_R = np.divide(1.0, R, out=np.zeros_like(R), where=R != 0.0)
    sin_t = np.sin(theta)
    return (
        np.where(valid, X * inv_R * sin_t, 0.0),
        np.where(valid, Y * inv_R * sin_t, 0.0),
        np.where(valid, -np.cos(theta), 1.0),
        valid,
    )


def _lut_project(dx, dy, dz, *, width, height, fx, fy, cx, cy, k, theta_max):
    """Unit view direction -> NDC."""
    import numpy as np  # noqa: PLC0415

    sin_t = np.hypot(dx, dy)
    theta = np.arctan2(sin_t, -dz)  # -dz so theta > pi/2 stays representable
    r = _theta_d_extended(theta, k, theta_max)
    scale = np.divide(r, sin_t, out=np.zeros_like(r), where=sin_t != 0.0)
    return cx / width + dx * scale * fx / width, cy / height - dy * scale * fy / height


def _lut_exit_table(*, tex_w, tex_h, **geom):
    """Octahedral direction map -> NDC.

    Octahedral decoding puts +Z at the texture centre while the camera looks -Z,
    so Z is flipped: that parks the optical axis at the centre and pushes the
    encoding's unavoidable opposite-pole seam out to the texture edges, where no
    real ray lands.
    """
    dx, dy, dz = _oct_to_unit_vector(_texel_centers(tex_w) * 2.0 - 1.0, _texel_centers(tex_h) * 2.0 - 1.0)
    return _lut_project(dx, dy, -dz, **geom)


def author_lut_fisheye(
    prim,
    intrinsic: Mapping[str, Any],
    width: int,
    height: int,
    *,
    max_fov_deg: float,
    out_dir: "Path | str",
    name: str,
    tex_scale: float = DEFAULT_LENS_LUT_SCALE,
) -> "dict[str, Any]":
    """Bake this lens into ``lut`` tables and apply ``OmniLensDistortionLutAPI``.

    Use this for a fisheye whose real field exceeds 180 deg;
    :func:`author_opencv_fisheye` cannot render one. Same Kannala-Brandt
    coefficients, same optical centre — the LUT simply carries them all the way
    out to ``max_fov_deg / 2`` instead of stopping at the hemisphere. On the
    ~221 deg head lenses that takes the traced image circle from r = 817 px to
    r = 1001 px and the fraction of the frame holding real geometry from 69.9 %
    to 91.8 % (measured against the factory calibration photos, whose image
    circles sit at r = 1000-1010 px).

    The two EXR tables are written under ``<out_dir>/lens/`` and referenced by a
    path relative to ``render_layer.usda``, so the scene directory stays
    relocatable.

    Raises rather than falling back to ``opencvFisheye`` on failure, and that is
    deliberate: ``render_node.cpp`` sizes its black mask from the scene YAML's
    ``mask_fov_deg`` on the assumption that a >180 deg lens really was authored
    as a LUT. A silent fallback would leave the mask at the true radius while
    the renderer only filled the 180 deg disc, ringing every frame with
    untraced environment colour — a wrong image that still looks plausible.
    Failing here instead makes the scene refuse to assemble.

    Returns a small dict describing what was baked, for the caller to log.
    """
    import numpy as np  # noqa: PLC0415
    from pxr import Sdf, Gf  # noqa: PLC0415

    out_dir = Path(out_dir)
    k = np.array(
        [float(intrinsic.get(key, 0.0)) for key in ("k1", "k2", "k3", "k4")],
        dtype=np.float64,
    )
    theta_max = np.radians(float(max_fov_deg) / 2.0)
    geom = dict(
        width=int(width),
        height=int(height),
        fx=float(intrinsic.get("fx", 0.0)),
        fy=float(intrinsic.get("fy", 0.0)),
        cx=float(intrinsic.get("cx", 0.0)),
        cy=float(intrinsic.get("cy", 0.0)),
        k=k,
        theta_max=float(theta_max),
    )
    if geom["fx"] <= 0.0 or geom["fy"] <= 0.0:
        raise ValueError(f"lens LUT for {name!r} needs positive fx/fy, got {geom['fx']} / {geom['fy']}")

    tex_w = max(int(round(int(width) * tex_scale)), 2)
    tex_h = max(int(round(int(height) * tex_scale)), 2)

    dx, dy, dz, valid = _lut_enter_table(tex_w=tex_w, tex_h=tex_h, **geom)
    px, py = _lut_exit_table(tex_w=tex_w, tex_h=tex_h, **geom)

    lens_dir = out_dir / LENS_LUT_SUBDIR
    enter = _write_exr(lens_dir / f"{name}_rayEnterDirection.exr", {"R": dx, "G": dy, "B": dz})
    exit_ = _write_exr(lens_dir / f"{name}_rayExitPosition.exr", {"R": px, "G": py})

    schema = Sdf.TokenListOp()
    schema.prependedItems = ["OmniLensDistortionLutAPI"]
    prim.SetMetadata("apiSchemas", schema)

    pfx = "omni:lensdistortion:lut"
    prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("lut")
    prim.CreateAttribute(f"{pfx}:rayEnterDirectionTexture", Sdf.ValueTypeNames.Asset).Set(
        f"./{LENS_LUT_SUBDIR}/{enter.name}"
    )
    prim.CreateAttribute(f"{pfx}:rayExitPositionTexture", Sdf.ValueTypeNames.Asset).Set(
        f"./{LENS_LUT_SUBDIR}/{exit_.name}"
    )
    prim.CreateAttribute(f"{pfx}:nominalWidth", Sdf.ValueTypeNames.Float).Set(float(width))
    prim.CreateAttribute(f"{pfx}:nominalHeight", Sdf.ValueTypeNames.Float).Set(float(height))
    prim.CreateAttribute(f"{pfx}:opticalCenter", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(geom["cx"], geom["cy"]))

    return {
        "name": name,
        "texture": [tex_w, tex_h],
        "max_fov_deg": float(max_fov_deg),
        "radius_px": float(geom["fx"] * _theta_d(np.array(theta_max), k)),
        "valid_fraction": float(valid.mean()),
        "bytes": enter.stat().st_size + exit_.stat().st_size,
    }


def author_fisheye_lens(
    prim,
    *,
    cam_cfg: Mapping[str, Any],
    intrinsic: Mapping[str, Any],
    width: int,
    height: int,
    out_dir: "Path | str",
    name: str,
) -> str:
    """Author whichever fisheye lens schema this camera's field of view calls for.

    Kept here rather than in each assembler on purpose: the choice between the
    built-in ``opencvFisheye`` and a baked ``lut`` is exactly the kind of
    camera-level detail that used to be duplicated per assembler and drift, and
    getting it wrong in one of them is invisible until someone compares two
    renders side by side.

    ``<= 180 deg`` (or no declared FOV) keeps the built-in model, which is
    cheaper and needs no side-car files. Past 180 deg the built-in model
    silently stops at the hemisphere, so the lens is baked into LUT tables
    instead — see :func:`author_lut_fisheye`.

    Returns a one-line summary for the caller to log.
    """
    fov = cam_cfg.get("max_fov_deg", cam_cfg.get("mask_fov_deg"))
    try:
        max_fov_deg = float(fov)
    except (TypeError, ValueError):
        max_fov_deg = 0.0

    if max_fov_deg <= 180.0:
        author_opencv_fisheye(prim, intrinsic, width, height)
        return f"opencvFisheye lens ({max_fov_deg or 'no'} deg declared FOV)"

    info = author_lut_fisheye(
        prim,
        intrinsic,
        width,
        height,
        max_fov_deg=max_fov_deg,
        out_dir=out_dir,
        name=name,
    )
    return (
        f"lut lens baked for {max_fov_deg} deg "
        f"(image circle r={info['radius_px']:.0f} px, "
        f"{info['texture'][0]}x{info['texture'][1]} tables, {info['bytes'] / 1e6:.1f} MB)"
    )


def author_camera_pose_from_extrinsic(prim, extrinsic: Optional[Mapping[str, Any]]) -> None:
    """Author a camera's local pose from a YAML ``extrinsic`` block, verbatim (no frame conversion).

    ``extrinsic.xyz`` (metres) + ``extrinsic.wxyz`` (unit quaternion, w-first) are written straight
    onto the prim as a Translate + Orient op stack, relative to the camera's parent xform. The
    values must already be in the USD/Isaac graphics camera convention (looks -Z, +Y up); this does
    NOT convert from the ROS/OpenCV optical frame. Missing / malformed values fall back to identity.
    """
    from pxr import UsdGeom, Gf

    extrinsic = extrinsic or {}
    xyz = extrinsic.get("xyz") or [0.0, 0.0, 0.0]
    wxyz = extrinsic.get("wxyz") or [1.0, 0.0, 0.0, 0.0]
    try:
        tx, ty, tz = float(xyz[0]), float(xyz[1]), float(xyz[2])
    except (TypeError, ValueError, IndexError):
        tx, ty, tz = 0.0, 0.0, 0.0
    try:
        qw, qx, qy, qz = float(wxyz[0]), float(wxyz[1]), float(wxyz[2]), float(wxyz[3])
    except (TypeError, ValueError, IndexError):
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
    xf.AddOrientOp().Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))


def synthesize_camera_intrinsics(
    cam_prim,
    *,
    fx: float,
    fy: float,
    width: int,
    height: int,
    min_range: float,
    max_range: float,
    h_aperture: float = DEFAULT_HORIZONTAL_APERTURE,
) -> None:
    """Author focalLength / aperture / clippingRange from pixel intrinsics.

    USD's ``UsdGeom.Camera`` relates focal/aperture to pixel intrinsics via::

        fx_pixels = focalLength * (width / horizontalAperture)
        fy_pixels = focalLength * (height / verticalAperture)

    ``horizontalAperture`` is pinned to ``h_aperture`` so the rendered fx matches the YAML fx
    exactly; ``verticalAperture`` follows from fy so non-square pixels (fx != fy) are preserved.
    These USD attributes only steer the renderer's image-plane projection — the published
    CameraInfo is built from the manifest's verbatim YAML intrinsics, not from here.
    """
    from pxr import Sdf, Gf

    focal_length = fx * h_aperture / max(width, 1)
    v_aperture = h_aperture * (height / max(width, 1)) * (fx / max(fy, 1e-6))
    cam_prim.CreateAttribute("focalLength", Sdf.ValueTypeNames.Float).Set(float(focal_length))
    cam_prim.CreateAttribute("horizontalAperture", Sdf.ValueTypeNames.Float).Set(float(h_aperture))
    cam_prim.CreateAttribute("verticalAperture", Sdf.ValueTypeNames.Float).Set(float(v_aperture))
    cam_prim.CreateAttribute("clippingRange", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(min_range, max_range))


def define_render_product(stage, *, rp_path: str, cam_path: str, width: int, height: int, want_depth: bool):
    """Define a ``RenderProduct`` at ``rp_path`` targeting ``cam_path``.

    Authors an ``LdrColor`` RenderVar (colour) and, when ``want_depth``, a
    ``DistanceToImagePlaneSD`` RenderVar (depth), wires ``orderedVars``, and sets
    ``resolution``, ``omni:rtx:rendermode`` and the RTPT path-splitting knobs.
    Returns the RenderProduct prim.
    """
    from pxr import Sdf, Gf

    rp_prim = stage.DefinePrim(rp_path, "RenderProduct")
    rp_prim.CreateRelationship("camera").SetTargets([Sdf.Path(cam_path)])

    rv_path = f"{rp_path}/LdrColor"
    rv_prim = stage.DefinePrim(rv_path, "RenderVar")
    rv_prim.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("LdrColor")

    ordered_var_targets = [Sdf.Path(rv_path)]
    if want_depth:
        depth_rv_path = f"{rp_path}/DistanceToImagePlaneSD"
        depth_rv_prim = stage.DefinePrim(depth_rv_path, "RenderVar")
        depth_rv_prim.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("DistanceToImagePlaneSD")
        ordered_var_targets.append(Sdf.Path(depth_rv_path))

    rp_prim.CreateRelationship("orderedVars").SetTargets(ordered_var_targets)
    rp_prim.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(width, height))
    rp_prim.CreateAttribute("omni:rtx:rendermode", Sdf.ValueTypeNames.Token).Set("RealTimePathTracing")
    # Path splitting off: the one RTPT knob that measurably buys frame time (-20%).
    # It costs flatter contact shadows / ambient occlusion, most visible on the
    # head-front and stereo cameras. Bounce counts, the denoiser toggles and spp were
    # all measured and change nothing here, so do not re-add them.
    rp_prim.CreateAttribute("omni:rtx:rtpt:usePathSplitting", Sdf.ValueTypeNames.Bool).Set(False)
    rp_prim.CreateAttribute("omni:rtx:rtpt:splitGlass", Sdf.ValueTypeNames.Bool).Set(False)
    rp_prim.CreateAttribute("omni:rtx:rtpt:splitClearcoat", Sdf.ValueTypeNames.Bool).Set(False)
    rp_prim.CreateAttribute("omni:rtx:rtpt:splitRoughReflection", Sdf.ValueTypeNames.Bool).Set(False)
    return rp_prim


def define_free_cam(stage, *, width: int, height: int):
    """Author the FreeCam (GL viewport preset) camera + its RenderProduct.

    Fixed paths: camera at ``/RenderOVRTX/Cameras/FreeCam``, RenderProduct at
    ``/RenderOVRTX/FreeCam``. The camera is a wide 18mm/36mm preset at a fixed
    world pose with ``omni:resetXformStack`` so it ignores the parent hierarchy;
    the render node overwrites its transform at runtime. Returns the camera.
    """
    from pxr import Sdf, Gf, UsdGeom

    free_cam_path = "/RenderOVRTX/Cameras/FreeCam"
    free_cam = UsdGeom.Camera.Define(stage, free_cam_path)
    free_cam.GetPrim().CreateAttribute("focalLength", Sdf.ValueTypeNames.Float).Set(18.0)
    free_cam.GetPrim().CreateAttribute("horizontalAperture", Sdf.ValueTypeNames.Float).Set(36.0)
    free_cam.GetPrim().CreateAttribute("verticalAperture", Sdf.ValueTypeNames.Float).Set(36.0 * height / max(width, 1))
    free_cam.GetPrim().CreateAttribute("clippingRange", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.01, 10000000))
    free_xf = UsdGeom.Xformable(free_cam.GetPrim())
    free_xf.AddTransformOp().Set(Gf.Matrix4d(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 2, 1))
    free_cam.GetPrim().CreateAttribute("omni:resetXformStack", Sdf.ValueTypeNames.Bool).Set(True)

    free_rp_path = "/RenderOVRTX/FreeCam"
    free_rp_prim = stage.DefinePrim(free_rp_path, "RenderProduct")
    free_rp_prim.CreateRelationship("camera").SetTargets([Sdf.Path(free_cam_path)])
    free_rv_path = f"{free_rp_path}/LdrColor"
    free_rv_prim = stage.DefinePrim(free_rv_path, "RenderVar")
    free_rv_prim.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("LdrColor")
    free_rp_prim.CreateRelationship("orderedVars").SetTargets([Sdf.Path(free_rv_path)])
    free_rp_prim.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(width, height))
    free_rp_prim.CreateAttribute("omni:rtx:rendermode", Sdf.ValueTypeNames.Token).Set("RealTimePathTracing")
    return free_cam


__all__ = [
    "DEFAULT_HORIZONTAL_APERTURE",
    "DEFAULT_LENS_LUT_SCALE",
    "LENS_LUT_SUBDIR",
    "author_opencv_fisheye",
    "author_opencv_pinhole",
    "author_lut_fisheye",
    "author_fisheye_lens",
    "author_camera_pose_from_extrinsic",
    "synthesize_camera_intrinsics",
    "define_render_product",
    "define_free_cam",
]
