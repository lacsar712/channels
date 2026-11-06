"""
Consumer-session runtime policy stack.

This module implements a *session-scoped* runtime policy stack that is
independent of:

* the permanent ``CHANNEL_LAYERS`` backend selection and its capacity/timeout
  configuration (it never mutates them),
* routing (``URLRouter`` compilation and host matching),
* the plain ASGI server lifecycle, and
* the Django authentication middleware loading.

The stack is bound to the current cooperative scope through
:mod:`contextvars`. Every consumer application instance initialises a fresh
stack when it starts, so two consumers running concurrently on the same event
loop each read their own stack and can never cross. Code nested inside a
handler (sub-callbacks, helpers, hooks) always reads the *live* stack at the
moment an outbound write, an accept or a dispatch admission happens -- values
are never snapshotted when a handler frame is entered.

Each stack frame carries three independent gates:

``dispatch_admit``
    Controls whether the *next* inbound message is admitted into consumer
    processing.

    * ``honor_cfg`` -- neutral; existing behaviour (and channel layer config
      driven errors such as ``ChannelFull``) is preserved.
    * ``deny_new`` -- every *new* inbound dispatch observed by the frame is
      observably rejected. The processing already in flight when the frame is
      pushed is neither cancelled nor rewritten into an empty success; only
      dispatches admitted afterwards are constrained.
    * ``one_more`` -- exactly one more inbound dispatch is admitted
      successfully during the frame's lifetime; the following ones are
      rejected. The decision is recomputed against the live stack top for
      every inbound message.

``outbound_gate``
    Controls new writes towards the channel layer or the socket.

    * ``honor_cfg`` -- neutral; existing behaviour is preserved.
    * ``deny_new`` -- every new ``channel_layer.send`` /
      ``channel_layer.group_send`` and consumer socket write is observably
      rejected. A write already in flight when the frame is pushed is not cut
      off mid-frame and faked as complete.
    * ``force_drop`` -- every requested write immediately takes the failure
      surface (it raises); nothing is enqueued and delivery is never faked.

``lifecycle_gate``
    Controls WebSocket accept/close.

    * ``honor_cfg`` -- neutral; existing accept/close behaviour is preserved.
    * ``deny_accept`` -- new ``accept()`` calls are observably rejected and
      never silently marked as accepted.
    * ``force_close`` -- when the frame pops (normally or while an exception
      unwinds), every session accepted while the frame was active and still
      open is closed with an observable disconnect. Sessions accepted outside
      the frame (including neighbouring consumer sessions) are never touched.

Frames can be pushed/popped explicitly (``push``/``frame.pop``) or used as
sync/async context managers. Popping an inner frame restores the outer view.
With an empty stack, behaviour is identical to the existing API.
"""

from __future__ import annotations

import contextvars
import logging

logger = logging.getLogger("channels.policies")


# ---------------------------------------------------------------------------
# Gate values
# ---------------------------------------------------------------------------

#: Neutral value; defer to existing configuration/behaviour.
HONOR_CFG = "honor_cfg"

#: Reject new inbound dispatches / outbound writes.
DENY_NEW = "deny_new"

#: Allow exactly one more inbound dispatch.
ONE_MORE = "one_more"

#: Make every requested outbound write fail/drop observably.
FORCE_DROP = "force_drop"

#: Reject new websocket accept attempts.
DENY_ACCEPT = "deny_accept"

#: Close sessions accepted under the frame when the frame pops.
FORCE_CLOSE = "force_close"

_DISPATCH_ADMIT_VALUES = frozenset({HONOR_CFG, DENY_NEW, ONE_MORE})
_OUTBOUND_GATE_VALUES = frozenset({HONOR_CFG, DENY_NEW, FORCE_DROP})
_LIFECYCLE_GATE_VALUES = frozenset({HONOR_CFG, DENY_ACCEPT, FORCE_CLOSE})

#: Close code emitted when a ``force_close`` frame pops.
FORCE_CLOSE_CODE = 1000

# Inbound message types that are part of the connection lifecycle rather than
# consumer work dispatches. They are always admitted so that sessions can
# still close/clean up while a dispatch_admit gate is active.
_LIFECYCLE_INBOUND_TYPES = frozenset({"websocket.disconnect", "http.disconnect"})


# ---------------------------------------------------------------------------
# Observable failure surfaces
# ---------------------------------------------------------------------------


class PolicyError(Exception):
    """Base class for observable session-policy failures."""


class OutboundWriteDenied(PolicyError):
    """Raised when an ``outbound_gate=deny_new`` frame rejects a new write."""

    def __init__(self, target):
        self.target = target
        super().__init__(
            "Outbound write denied by consumer session policy: %r" % (target,)
        )


class OutboundWriteDropped(PolicyError):
    """Raised when an ``outbound_gate=force_drop`` frame fails a new write."""

    def __init__(self, target):
        self.target = target
        super().__init__(
            "Outbound write force-dropped by consumer session policy: %r"
            % (target,)
        )


class AcceptDenied(PolicyError):
    """Raised when a ``lifecycle_gate=deny_accept`` frame rejects accept()."""


# ---------------------------------------------------------------------------
# Internal per-session state
# ---------------------------------------------------------------------------


class _SessionState:
    """
    Policy-owned state for one running consumer session.

    This lives in a context variable owned by this module; no policy state is
    attached to the consumer, the channel layer, the router or any task.
    """

    __slots__ = ("accepted", "ended", "async_close", "sync_close")

    def __init__(self, async_close=None, sync_close=None):
        self.accepted = False
        self.ended = False
        # Callables that emit a raw websocket.close for this session. They
        # bypass the outbound gate because close is lifecycle, not data.
        self.async_close = async_close
        self.sync_close = sync_close


_stack_var = contextvars.ContextVar("channels_policy_stack")
_session_var = contextvars.ContextVar("channels_policy_session")


# ---------------------------------------------------------------------------
# A single stack frame
# ---------------------------------------------------------------------------


class PolicyFrame:
    """
    One frame of the consumer-session runtime policy stack.

    Instances are produced by :func:`push`. They are also context managers
    (both sync and async); ``push()`` enters the frame immediately so a frame
    may stay active across several inbound dispatches and be popped later from
    a different handler.
    """

    def __init__(
        self,
        dispatch_admit=HONOR_CFG,
        outbound_gate=HONOR_CFG,
        lifecycle_gate=HONOR_CFG,
        on_event=None,
    ):
        if dispatch_admit not in _DISPATCH_ADMIT_VALUES:
            raise ValueError(
                "dispatch_admit must be one of %r, got %r"
                % (sorted(_DISPATCH_ADMIT_VALUES), dispatch_admit)
            )
        if outbound_gate not in _OUTBOUND_GATE_VALUES:
            raise ValueError(
                "outbound_gate must be one of %r, got %r"
                % (sorted(_OUTBOUND_GATE_VALUES), outbound_gate)
            )
        if lifecycle_gate not in _LIFECYCLE_GATE_VALUES:
            raise ValueError(
                "lifecycle_gate must be one of %r, got %r"
                % (sorted(_LIFECYCLE_GATE_VALUES), lifecycle_gate)
            )
        self.dispatch_admit = dispatch_admit
        self.outbound_gate = outbound_gate
        self.lifecycle_gate = lifecycle_gate
        self.on_event = on_event

        # Observable counters / event log.
        self.inbound_denied = 0
        self.outbound_denied = 0
        self.outbound_dropped = 0
        self.accept_denied = 0
        self.force_closed = 0
        self.events = []

        # Remaining admissions for a one_more frame (exactly one).
        self._admit_remaining = 1 if dispatch_admit == ONE_MORE else 0
        # Sessions accepted while this frame was the active policy owner.
        self._sessions = set()
        self._entered = False
        self.depth = None

    # -- observation -------------------------------------------------------

    @property
    def one_more_remaining(self):
        """Remaining one_more admissions (0 if this is not a one_more frame)."""
        return self._admit_remaining if self.dispatch_admit == ONE_MORE else None

    def _record(self, kind, **detail):
        event = {"kind": kind}
        event.update(detail)
        self.events.append(event)
        if kind == "inbound_denied":
            self.inbound_denied += 1
        elif kind == "outbound_denied":
            self.outbound_denied += 1
        elif kind == "outbound_dropped":
            self.outbound_dropped += 1
        elif kind == "accept_denied":
            self.accept_denied += 1
        elif kind == "force_closed":
            self.force_closed += 1
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                logger.exception("policy frame on_event callback failed")
        logger.warning(
            "session policy frame #%s blocked %s: %s", self.depth, kind, detail
        )
        return event

    # -- push / pop --------------------------------------------------------

    def enter(self):
        if self._entered:
            raise RuntimeError("policy frame is already pushed")
        stack = _stack_var.get(())
        self.depth = len(stack)
        _stack_var.set(stack + (self,))
        self._entered = True
        return self

    def _detach(self):
        if not self._entered:
            raise RuntimeError("policy frame has already been popped")
        stack = _stack_var.get(())
        if not stack or stack[-1] is not self:
            raise RuntimeError("policy frames must be popped in push order")
        parent = stack[:-1]
        sessions = self._sessions
        self._sessions = set()
        self._entered = False
        _stack_var.set(parent)
        return parent, sessions

    def pop(self):
        """Pop the frame synchronously (use inside sync consumers/sections)."""
        parent, sessions = self._detach()
        if self.lifecycle_gate == FORCE_CLOSE:
            for session in list(sessions):
                if not session.ended:
                    self._force_close_sync(session)
        self._bubble(sessions, parent)

    async def apop(self):
        """Pop the frame asynchronously (use inside async consumers/sections)."""
        parent, sessions = self._detach()
        if self.lifecycle_gate == FORCE_CLOSE:
            for session in list(sessions):
                if not session.ended:
                    await self._force_close_async(session)
        self._bubble(sessions, parent)

    def _bubble(self, sessions, parent):
        # Sessions still open when an inner frame leaves become owned by the
        # immediate outer frame so an outer force_close still covers accepts
        # that happened within its dynamic extent.
        if parent:
            owner = parent[-1]
            for session in sessions:
                if not session.ended:
                    owner._sessions.add(session)

    def _force_close_sync(self, session):
        # Record before emitting so observers never see the observable close
        # ahead of the frame accounting.
        session.ended = True
        self._record("force_closed")
        try:
            if session.sync_close is None:
                raise RuntimeError(
                    "lifecycle_gate=force_close popped synchronously in an "
                    "async consumer session; use 'async with' / await "
                    "frame.apop() there"
                )
            session.sync_close(FORCE_CLOSE_CODE)
        except BaseException:
            logger.exception("force_close failed to close a session")

    async def _force_close_async(self, session):
        session.ended = True
        self._record("force_closed")
        try:
            if session.async_close is None:
                raise RuntimeError(
                    "lifecycle_gate=force_close popped asynchronously in a "
                    "sync consumer session; use 'with' / frame.pop() there"
                )
            await session.async_close(FORCE_CLOSE_CODE)
        except BaseException:
            logger.exception("force_close failed to close a session")

    # -- context manager protocols ----------------------------------------

    def __enter__(self):
        # push() already entered the frame when used as ``with push()``.
        return self if self._entered else self.enter()

    def __exit__(self, exc_type, exc, tb):
        if self._entered:
            self.pop()
        return False

    async def __aenter__(self):
        return self if self._entered else self.enter()

    async def __aexit__(self, exc_type, exc, tb):
        if self._entered:
            await self.apop()
        return False


def push(
    *,
    dispatch_admit=HONOR_CFG,
    outbound_gate=HONOR_CFG,
    lifecycle_gate=HONOR_CFG,
    on_event=None,
):
    """
    Push a new frame onto the current consumer-session policy stack and
    return it. Pop it later with :meth:`PolicyFrame.pop` (sync) or
    :meth:`PolicyFrame.apop` (async), or use the returned frame as a
    ``with`` / ``async with`` context manager.
    """
    return PolicyFrame(
        dispatch_admit=dispatch_admit,
        outbound_gate=outbound_gate,
        lifecycle_gate=lifecycle_gate,
        on_event=on_event,
    ).enter()


# ---------------------------------------------------------------------------
# Session binding (used by the consumer implementation)
# ---------------------------------------------------------------------------


def bind_session(*, async_close=None, sync_close=None):
    """Bind fresh policy session state for a starting consumer instance."""
    session = _SessionState(async_close=async_close, sync_close=sync_close)
    return _session_var.set(session)


def end_session(token):
    """Detach the policy session state for a finished consumer instance."""
    session = _session_var.get(None)
    if session is not None:
        session.ended = True
    _session_var.reset(token)


def reset_stack():
    """
    Start each consumer session with an empty stack. Returns a reset token so
    the enclosing scope (e.g. inline ASGI wrapping) can be restored.
    """
    return _stack_var.set(())


def restore_stack(token):
    _stack_var.reset(token)


# ---------------------------------------------------------------------------
# Gate decisions (called from the consumer integration points)
# ---------------------------------------------------------------------------


def _frames():
    return _stack_var.get(())


def admit_inbound(message):
    """
    Merge the live stack to decide whether one inbound message may enter
    consumer processing now. Returns ``True`` to admit, ``False`` to
    observably reject (the caller drops the message and keeps serving).

    The view is computed on every call from the current stack; a one_more
    token is consumed only when the message is actually admitted, so every
    subsequent message re-evaluates against the then-current stack top.
    """
    frames = _frames()
    if not frames:
        return True

    # Lifecycle teardown messages must always get through.
    message_type = message.get("type") if isinstance(message, dict) else None
    if message_type in _LIFECYCLE_INBOUND_TYPES:
        return True

    denials = []
    one_more_frames = []
    for frame in frames:
        if frame.dispatch_admit == DENY_NEW:
            denials.append((frame, "deny_new"))
        elif frame.dispatch_admit == ONE_MORE:
            one_more_frames.append(frame)

    if not denials:
        exhausted = [f for f in one_more_frames if f._admit_remaining <= 0]
        if exhausted:
            denials = [(f, "one_more_exhausted") for f in exhausted]
        else:
            # Merged admission: consume one token on every active one_more
            # frame; honor_cfg frames stay neutral.
            for frame in one_more_frames:
                frame._admit_remaining -= 1
            return True

    for frame, reason in denials:
        frame._record("inbound_denied", reason=reason, type=message_type)
    return False


def guard_outbound(target):
    """
    Merge the live stack for one new outbound write. Raises
    :class:`OutboundWriteDropped` (force_drop wins across frames) or
    :class:`OutboundWriteDenied` (deny_new) before any write happens; with
    honor_cfg everywhere this returns and the existing write proceeds.
    """
    frames = _frames()
    if not frames:
        return
    drops = [f for f in frames if f.outbound_gate == FORCE_DROP]
    if drops:
        for frame in drops:
            frame._record("outbound_dropped", **target)
        raise OutboundWriteDropped(target)
    denies = [f for f in frames if f.outbound_gate == DENY_NEW]
    if denies:
        for frame in denies:
            frame._record("outbound_denied", **target)
        raise OutboundWriteDenied(target)


def guard_accept():
    """
    Reject a new websocket accept while any frame denies it. The accept
    message must not be emitted and the session must not be marked accepted.
    """
    frames = _frames()
    denied = [f for f in frames if f.lifecycle_gate == DENY_ACCEPT]
    if denied:
        for frame in denied:
            frame._record("accept_denied")
        raise AcceptDenied("websocket.accept denied by session lifecycle policy")


def note_accepted():
    """
    Record a successful accept: the owning frame (stack top) becomes
    responsible for the session, e.g. for force_close on pop.
    """
    session = _session_var.get(None)
    if session is None:
        return
    session.accepted = True
    frames = _frames()
    if frames:
        frames[-1]._sessions.add(session)


def note_close_requested():
    """An explicit close has been emitted; the session is ending."""
    session = _session_var.get(None)
    if session is not None:
        session.ended = True


def note_session_ended():
    """The peer went away (websocket.disconnect); the session is over."""
    session = _session_var.get(None)
    if session is not None:
        session.ended = True


# ---------------------------------------------------------------------------
# Introspection: the merged view of configuration + current call
# ---------------------------------------------------------------------------


class PolicyView:
    """
    Immutable merged view of the live stack for the current scope.

    ``honor_cfg`` everywhere (or an empty stack) reports the same permissions
    the existing API grants without any frame pushed.
    """

    __slots__ = (
        "depth",
        "dispatch_admit",
        "inbound_open",
        "one_more_remaining",
        "outbound_gate",
        "lifecycle_gate",
        "force_close_pending",
    )

    def __init__(
        self,
        depth,
        dispatch_admit,
        inbound_open,
        one_more_remaining,
        outbound_gate,
        lifecycle_gate,
        force_close_pending,
    ):
        self.depth = depth
        self.dispatch_admit = dispatch_admit
        self.inbound_open = inbound_open
        self.one_more_remaining = one_more_remaining
        self.outbound_gate = outbound_gate
        self.lifecycle_gate = lifecycle_gate
        self.force_close_pending = force_close_pending

    def __repr__(self):
        return (
            "PolicyView(depth=%d, dispatch_admit=%s, inbound_open=%s, "
            "one_more_remaining=%r, outbound_gate=%s, lifecycle_gate=%s, "
            "force_close_pending=%s)"
            % (
                self.depth,
                self.dispatch_admit,
                self.inbound_open,
                self.one_more_remaining,
                self.outbound_gate,
                self.lifecycle_gate,
                self.force_close_pending,
            )
        )

    def __eq__(self, other):
        return isinstance(other, PolicyView) and all(
            getattr(self, name) == getattr(other, name) for name in self.__slots__
        )


def current_view():
    """Return the merged :class:`PolicyView` for the current cooperative scope."""
    frames = _frames()
    depth = len(frames)

    # Inbound axis.
    dispatch_admit = HONOR_CFG
    one_more_remaining = None
    if any(f.dispatch_admit == DENY_NEW for f in frames):
        dispatch_admit = DENY_NEW
    else:
        one_more_frames = [f for f in frames if f.dispatch_admit == ONE_MORE]
        if one_more_frames:
            dispatch_admit = ONE_MORE
            one_more_remaining = min(f._admit_remaining for f in one_more_frames)
    inbound_open = dispatch_admit == HONOR_CFG or (
        dispatch_admit == ONE_MORE and one_more_remaining > 0
    )

    # Outbound axis (force_drop wins).
    if any(f.outbound_gate == FORCE_DROP for f in frames):
        outbound_gate = FORCE_DROP
    elif any(f.outbound_gate == DENY_NEW for f in frames):
        outbound_gate = DENY_NEW
    else:
        outbound_gate = HONOR_CFG

    # Lifecycle axis.
    lifecycle_gate = (
        DENY_ACCEPT
        if any(f.lifecycle_gate == DENY_ACCEPT for f in frames)
        else HONOR_CFG
    )
    force_close_pending = any(f.lifecycle_gate == FORCE_CLOSE for f in frames)

    return PolicyView(
        depth=depth,
        dispatch_admit=dispatch_admit,
        inbound_open=inbound_open,
        one_more_remaining=one_more_remaining,
        outbound_gate=outbound_gate,
        lifecycle_gate=lifecycle_gate,
        force_close_pending=force_close_pending,
    )


# ---------------------------------------------------------------------------
# Policy-aware channel layer view (outbound axis only)
# ---------------------------------------------------------------------------


class SessionChannelLayer:
    """
    Transparent policy-aware view of a real channel layer backend, held by a
    consumer for the duration of its session.

    Only the outbound ``send`` / ``group_send`` entry points consult the
    session policy stack; every other attribute (``receive``,
    ``new_channel``, ``group_add``, ...) delegates to the wrapped backend
    unchanged. The backend itself, its name and its capacity/timeout
    configuration are never modified.
    """

    __slots__ = ("_backend",)

    def __init__(self, backend):
        self._backend = backend

    async def send(self, channel, message):
        guard_outbound(
            {
                "via": "channel_layer.send",
                "channel": channel,
                "type": message.get("type") if isinstance(message, dict) else None,
            }
        )
        return await self._backend.send(channel, message)

    async def group_send(self, group, message):
        guard_outbound(
            {
                "via": "channel_layer.group_send",
                "group": group,
                "type": message.get("type") if isinstance(message, dict) else None,
            }
        )
        return await self._backend.group_send(group, message)

    def __getattr__(self, item):
        return getattr(self._backend, item)


def gate_channel_layer(backend):
    """Return the policy-aware view of ``backend`` (or ``None``)."""
    if backend is None:
        return None
    return SessionChannelLayer(backend)
