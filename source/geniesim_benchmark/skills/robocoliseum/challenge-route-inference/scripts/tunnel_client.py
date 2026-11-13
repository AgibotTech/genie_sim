# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""tunnel_client —— 隧道协议客户端(不依赖 openpi / jax / cv2,可单独测试)。

从 tunnel_agent.py 抽出:数据帧编解码、状态机、TunnelClient。策略加载与 obs 适配留在
tunnel_agent.py。
"""
from __future__ import annotations

import asyncio
import enum
import json
import logging
import struct
import sys
import time
import traceback
import uuid
from typing import Awaitable, Callable, Set, Union
from urllib.parse import urlencode, urlparse, urlunparse

import websockets
from websockets.asyncio.client import connect as ws_connect

logger = logging.getLogger("tunnel_agent")


# ---------------------------------------------------------------------------
# 数据帧编解码 —— 与 Go pkg/tunnel 逐字节兼容
# 布局: [uint32 BE sessionID 长度][sessionID 字节][payload 字节]
# ---------------------------------------------------------------------------

MAX_SESSION_ID_LEN = 256


def encode_data_frame(session_id: str, payload: bytes) -> bytes:
    sid = session_id.encode("utf-8")
    if len(sid) > MAX_SESSION_ID_LEN:
        raise ValueError(
            f"session_id exceeds {MAX_SESSION_ID_LEN} bytes: got {len(sid)}"
        )
    return struct.pack(">I", len(sid)) + sid + payload


def decode_data_frame(frame: bytes) -> tuple[str, bytes]:
    if len(frame) < 4:
        raise ValueError("tunnel: data frame malformed (too short)")
    (sid_len,) = struct.unpack(">I", frame[:4])
    if sid_len > len(frame) - 4:
        raise ValueError("tunnel: data frame malformed (sid_len out of bounds)")
    return frame[4 : 4 + sid_len].decode("utf-8"), bytes(frame[4 + sid_len :])


# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------


class State(str, enum.Enum):
    QUEUED = "QUEUED"
    WARMUP = "WARMUP"
    RUNNING = "RUNNING"
    DRAINING = "DRAINING"


class TunnelExhausted(RuntimeError):
    """run() 在达到 max_retries 后仍无法建立健康连接时抛出。"""


class TunnelRejected(RuntimeError):
    """握手被网关明确拒绝(401/403/404:token 无效、作业已终态或不属于该账号)。

    重试没有意义,直接退出并说明原因。"""


class WarmupFailed(RuntimeError):
    """warmup 阶段加载模型失败。继续连着只会让平台一直等一个永远不会 ready 的 agent。"""


class HandlerFailed(RuntimeError):
    """推理 handler 连续失败达到上限。"""


# 仿真器每次推理请求的超时(genie-sim CoRobotPolicy:120 秒,超时后重发,最多 30 次)。
# 这是一帧观测从发出到收到回包的总预算:下行传输 + 推理 + 回包都要落在里面。
SIM_INFER_TIMEOUT_SECONDS = 120
# 达到此大小就提示带宽风险(开启深度时单帧约 6.5MB)。
LARGE_FRAME_BYTES = 2_000_000

# 从最后一次有进展(收到 warmup / session_open / 数据帧)起,多久仍连不回来就放弃。
# 不能按次数算:网关要等读死线(最长 45 秒)才发现旧连接已死,在那之前同一 agent_id
# 的拨号都被当成重复连接直接关掉;按次数的预算几秒就耗尽了。握手被明确拒绝(401 等)
# 另有 TunnelRejected 立即退出,所以这里放宽不会让进程对着已结束的作业空转。
DEFAULT_RECONNECT_BUDGET_SECONDS = 180.0


# 握手成功后连接保持打开这么久才算「已连上」。网关对超过 parallelism 上限或重复的拨号
# 会在升级后立刻关闭连接,这段宽限期把这类拨号排除在外。
CONNECTED_GRACE_SECONDS = 3.0


# 这些 HTTP 状态说明「这次拨号本身不被接受」,而不是网络故障。
_FATAL_HANDSHAKE_STATUS = (401, 403, 404)


FrameHandler = Callable[[str, bytes], Union[bytes, Awaitable[bytes]]]


# ---------------------------------------------------------------------------
# TunnelClient —— 与 simubotix_agent.TunnelClient 等价的实现
# ---------------------------------------------------------------------------


class TunnelClient:
    def __init__(
        self,
        url: str,
        access_token: str,
        job_uuid: str,
        on_frame: FrameHandler,
        agent_id: str | None = None,
        on_session_close: Callable[[str], None] | None = None,
        pong_interval: float = 15.0,
        max_handler_errors: int = 5,
        describe_frame: Callable[[bytes], str] | None = None,
        on_connected: Callable[[], None] | None = None,
    ) -> None:
        self._url = url
        self._access_token = access_token
        self._job_uuid = job_uuid
        self.agent_id = agent_id or str(uuid.uuid4())
        self._on_frame = on_frame
        self._on_session_close = on_session_close
        self._on_connected = on_connected
        self.state: State = State.QUEUED
        self.state_history: list[State] = [State.QUEUED]
        self.active_sessions: Set[str] = set()
        self._drained = False
        self._inflight: Set[asyncio.Task] = set()
        self._progressed = False
        self._pong_interval = pong_interval
        self._max_handler_errors = max_handler_errors
        self._handler_errors = 0
        self._fatal: Exception | None = None
        self._describe_frame = describe_frame
        self._frame_reported = False
        self._largest_frame = 0
        self._last_frame_at: float | None = None

    def _build_ws_url(self) -> str:
        parts = urlparse(self._url)
        new_query = urlencode({"job": self._job_uuid, "agent": self.agent_id})
        if parts.query:
            new_query = parts.query + "&" + new_query
        return urlunparse(parts._replace(query=new_query))

    def _set_state(self, new_state: State) -> None:
        self.state = new_state
        self.state_history.append(new_state)
        logger.info("state -> %s", new_state.value)

    async def _invoke_handler(self, session_id: str, payload: bytes) -> bytes:
        # openpi 推理是同步、CPU/GPU 阻塞型,丢到工作线程跑,避免阻塞 recv 循环。
        return await asyncio.to_thread(self._on_frame, session_id, payload)

    async def run(
        self,
        reconnect_budget: float = DEFAULT_RECONNECT_BUDGET_SECONDS,
        initial_backoff: float = 0.5,
        max_backoff: float = 3.0,
    ) -> None:
        # max_backoff 取 3 秒:网关的重连窗口只有 30 秒,退避涨到 30 秒会越过窗口。
        attempt = 0
        last_progress = time.monotonic()
        while True:
            self._drained = False
            self._progressed = False
            try:
                await self.run_once()
            except websockets.exceptions.InvalidStatus as exc:
                status = exc.response.status_code
                if status in _FATAL_HANDSHAKE_STATUS:
                    raise TunnelRejected(
                        f"gateway rejected the dial with HTTP {status} "
                        f"(token invalid, job not active, or job not yours)"
                    ) from exc
                last_error: Exception = exc
                logger.warning("tunnel handshake failed: %s", exc)
            except (OSError, websockets.exceptions.WebSocketException) as exc:
                last_error = exc
                logger.warning("tunnel error: %s", exc)
            else:
                last_error = RuntimeError("WS closed without drain")
            # drain 优先于 _fatal:作业已正常收尾时,迟到的推理失败不该把退出码变成 1。
            if self._drained:
                return
            if self._fatal is not None:
                raise self._fatal
            # 本次连接收到过 warmup / session_open / 数据帧 = 链路是通的,之前的失败
            # 不该继续占用重连预算,否则长跑的 job 累计断线几次就会退出。
            if self._progressed:
                attempt = 0
                last_progress = time.monotonic()
            if time.monotonic() - last_progress >= reconnect_budget:
                raise TunnelExhausted(
                    f"no progress for {reconnect_budget:.0f}s after {attempt + 1} attempts: "
                    f"{last_error}"
                ) from last_error
            backoff = min(max_backoff, initial_backoff * (2 ** attempt))
            logger.info("reconnect in %.1fs (attempt %d)", backoff, attempt + 1)
            await asyncio.sleep(backoff)
            attempt += 1

    def _note_frame(self, nbytes: int, payload: bytes) -> None:
        """记录入站帧的可确定事实。不估传输速率:websockets 高层 API 看不到帧的真实
        传输耗时(从上次回包到收到下一帧的间隔含仿真器计算时间),据此推速率会误报。"""
        self._last_frame_at = time.monotonic()
        self._largest_frame = max(self._largest_frame, nbytes)
        if self._frame_reported or nbytes < LARGE_FRAME_BYTES:
            return
        self._frame_reported = True
        detail = ""
        if self._describe_frame is not None:
            try:
                detail = f" ({self._describe_frame(payload)})"
            except Exception:  # noqa: BLE001
                pass
        # 这是下限:120 秒里还要留给推理和回包,实际需要的带宽更高。
        need = nbytes * 8 / SIM_INFER_TIMEOUT_SECONDS / 1e6
        logger.warning(
            "observation frame: %.2f MB%s. The simulator waits at most %ds for each answer, "
            "and that budget also covers downloading this frame, so your downstream link to "
            "the gateway needs well over %.2f Mbit/s sustained (plus your inference time). "
            "On a slower link inference requests time out and the case fails: turn depth off "
            "if your model does not use it, or run this agent closer to the gateway. "
            "See challenge-diagnose-connection.",
            nbytes / 1e6, detail, SIM_INFER_TIMEOUT_SECONDS, need,
        )

    @staticmethod
    def _fmt_close(frame) -> str:
        return "none" if frame is None else str(frame.code)

    def _log_disconnect(self, exc, latency) -> None:
        since = (
            f"{time.monotonic() - self._last_frame_at:.1f}s"
            if self._last_frame_at is not None else "n/a"
        )
        logger.warning(
            "disconnected: rcvd=%s sent=%s in_flight=%d since_last_frame=%s "
            "largest_frame=%.2fMB keepalive_latency=%s. See challenge-diagnose-connection.",
            self._fmt_close(exc.rcvd), self._fmt_close(exc.sent), len(self._inflight),
            since, self._largest_frame / 1e6,
            "n/a" if latency is None else f"{latency:.2f}s",
        )

    async def _unsolicited_pong_loop(self, ws) -> None:
        # 下载多 MB 观测帧期间,网关的 ping 排在大帧后面到不了我们,我们也就没有 pong
        # 可回 —— 网关 45 秒读不到任何东西就判死。主动 pong 走上行方向,不受下行大帧
        # 影响,网关把任何 pong 都计为存活证据(job 1652 agent 1045:加上之前 14 次读超时,
        # 加上之后 0 次)。
        while True:
            await asyncio.sleep(self._pong_interval)
            try:
                await ws.pong()
            except websockets.exceptions.ConnectionClosed:
                return

    async def _connected_after_grace(self, ws) -> None:
        await asyncio.sleep(CONNECTED_GRACE_SECONDS)
        if self._on_connected is None or ws.close_code is not None:
            return
        try:
            self._on_connected()
        except Exception:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)

    async def run_once(self) -> None:
        url = self._build_ws_url()
        logger.info("dialing gateway: %s", url)
        async with ws_connect(
            url,
            additional_headers={"Authorization": f"Bearer {self._access_token}"},
            # 客户端不发 ping:保活由每 15 秒的主动 pong 完成。客户端 ping 要求网关回 pong,
            # 大帧在写时这次回写只能排队;旧版网关(写超时 5 秒)还会因此把连接写坏。
            # 也不设 ping_timeout:观测帧可达数 MB,pong 排在在途帧后面,客户端自己判死
            # 只会关掉一条健康的连接。死连接由网关的读死线发现,我们靠断线后重连恢复。
            ping_interval=None,
            ping_timeout=None,
            max_size=None,
        ) as ws:
            pong_task = asyncio.create_task(self._unsolicited_pong_loop(ws))
            connected_task = asyncio.create_task(self._connected_after_grace(ws))
            try:
                await self._recv_loop(ws)
            except websockets.exceptions.ConnectionClosedOK:
                return
            except websockets.exceptions.ConnectionClosed as exc:
                self._log_disconnect(exc, getattr(ws, "latency", None))
                raise
            finally:
                pong_task.cancel()
                connected_task.cancel()

    async def _recv_loop(self, ws) -> None:
        async for msg in ws:
            if isinstance(msg, str):
                await self._handle_control(ws, msg)
                if self.state == State.DRAINING:
                    await ws.close()
                    return
            elif isinstance(msg, (bytes, bytearray)):
                self._progressed = True
                self._spawn(self._handle_data(ws, bytes(msg)))

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)
        return task

    async def _handle_control(self, ws, msg: str) -> None:
        try:
            frame = json.loads(msg)
        except json.JSONDecodeError:
            return
        ctrl_type = frame.get("type")
        session_id = frame.get("session_id", "")

        if ctrl_type == "warmup":
            self._set_state(State.WARMUP)
            self._progressed = True
            try:
                await self._invoke_handler("", b"")
            except Exception as exc:  # noqa: BLE001
                logger.error("warmup failed: %s: %s", type(exc).__name__, exc)
                traceback.print_exc(file=sys.stderr)
                self._fatal = WarmupFailed(f"{type(exc).__name__}: {exc}")
                await ws.close(code=4001, reason="warmup failed")
                return
            await ws.send(json.dumps({"type": "ready"}))
            self._set_state(State.RUNNING)

        elif ctrl_type == "session_open":
            if session_id:
                self._progressed = True
                self.active_sessions.add(session_id)
                logger.info("session open: %s", session_id)

        elif ctrl_type == "session_close":
            if session_id:
                self.active_sessions.discard(session_id)
                if self._on_session_close is not None:
                    try:
                        self._on_session_close(session_id)
                    except Exception:  # noqa: BLE001
                        traceback.print_exc(file=sys.stderr)
                logger.info("session close: %s", session_id)

        elif ctrl_type == "drain":
            self.active_sessions.clear()
            self._drained = True
            self._set_state(State.DRAINING)

    async def _handle_data(self, ws, frame: bytes) -> None:
        try:
            session_id, payload = decode_data_frame(frame)
        except ValueError as exc:
            logger.warning("malformed data frame dropped: %s", exc)
            return
        self._note_frame(len(frame), payload)
        try:
            response = await self._invoke_handler(session_id, payload)
        except Exception:  # noqa: BLE001
            logger.error(
                "handler error on session=%s\n%s", session_id, traceback.format_exc()
            )
            # 不关连接:网关不会因此重发观测 —— 重发由仿真器在自己的推理超时(120 秒)后
            # 发起,与我们这边连接断没断无关,关连接只会白白多一次重连。这一帧就不回包,
            # 等仿真器超时重发。确定性的模型 bug 会一直失败,连续达到上限就退出。
            self._handler_errors += 1
            if self._handler_errors >= self._max_handler_errors:
                self._fatal = HandlerFailed(
                    f"inference failed {self._handler_errors} times in a row"
                )
                await ws.close(code=4000, reason="inference handler keeps failing")
            return
        self._handler_errors = 0
        try:
            await ws.send(encode_data_frame(session_id, response))
        except websockets.exceptions.ConnectionClosed:
            return
