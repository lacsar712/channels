"""
Tests for the consumer-session runtime policy stack (channels.policies).

Covers the three gates -- dispatch_admit, outbound_gate, lifecycle_gate --
the push/pop stack semantics, in-flight operation protection, cross-consumer
scope isolation on one event loop, nested frame merging, and the guarantee
that CHANNEL_LAYERS configuration and routing/auth are left untouched.
"""

import asyncio
from copy import deepcopy

import pytest
from django.test import override_settings

from channels import policies
from channels.consumer import AsyncConsumer
from channels.exceptions import StopConsumer
from channels.generic.websocket import (
    AsyncJsonWebsocketConsumer,
    AsyncWebsocketConsumer,
    JsonWebsocketConsumer,
    WebsocketConsumer,
)
from channels.layers import get_channel_layer
from channels.testing import ApplicationCommunicator, WebsocketCommunicator

FRESH_LAYER_SETTING = {
    "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}
}


# ---------------------------------------------------------------------------
# Pure stack mechanics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_stack_matches_config():
    view = policies.current_view()
    assert view.depth == 0
    assert view.dispatch_admit == policies.HONOR_CFG
    assert view.inbound_open is True
    assert view.one_more_remaining is None
    assert view.outbound_gate == policies.HONOR_CFG
    assert view.lifecycle_gate == policies.HONOR_CFG
    assert view.force_close_pending is False
    # Empty stack admits and does not guard.
    assert policies.admit_inbound({"type": "anything"}) is True
    policies.guard_outbound({"via": "test"})  # must not raise
    policies.guard_accept()  # must not raise


@pytest.mark.asyncio
async def test_push_pop_restores_merged_view():
    frame = policies.push(
        dispatch_admit=policies.DENY_NEW,
        outbound_gate=policies.FORCE_DROP,
        lifecycle_gate=policies.DENY_ACCEPT,
    )
    view = policies.current_view()
    assert view.depth == 1
    assert view.dispatch_admit == policies.DENY_NEW
    assert view.inbound_open is False
    assert view.outbound_gate == policies.FORCE_DROP
    assert view.lifecycle_gate == policies.DENY_ACCEPT
    frame.pop()
    assert policies.current_view().depth == 0
    assert policies.current_view().inbound_open is True


@pytest.mark.asyncio
async def test_invalid_gate_values_and_stack_order():
    with pytest.raises(ValueError):
        policies.push(dispatch_admit="nope")
    with pytest.raises(ValueError):
        policies.push(outbound_gate="nope")
    with pytest.raises(ValueError):
        policies.push(lifecycle_gate="nope")

    outer = policies.push(dispatch_admit=policies.DENY_NEW)
    inner = policies.push(outbound_gate=policies.DENY_NEW)
    # Frames must pop in LIFO order.
    with pytest.raises(RuntimeError):
        outer.pop()
    inner.pop()
    outer.pop()
    # A popped frame cannot pop twice.
    with pytest.raises(RuntimeError):
        outer.pop()


@pytest.mark.asyncio
async def test_one_more_live_reread_and_nested_merge():
    # Exactly one admission; the next call re-reads the (same) top and fails;
    # popping restores admission (no cached denial).
    frame = policies.push(dispatch_admit=policies.ONE_MORE)
    assert policies.admit_inbound({"type": "a"}) is True
    assert frame.one_more_remaining == 0
    assert policies.admit_inbound({"type": "b"}) is False
    assert policies.admit_inbound({"type": "c"}) is False
    assert frame.inbound_denied == 2
    frame.pop()
    assert policies.admit_inbound({"type": "d"}) is True

    # Merged view: a neutral inner honor_cfg frame does not weaken an outer
    # one_more frame; inner pop restores the outer state, outer pop restores
    # configuration behaviour.
    outer = policies.push(dispatch_admit=policies.ONE_MORE)
    inner = policies.push()  # all gates honor_cfg
    assert policies.admit_inbound({"type": "e"}) is True  # consumes outer token
    assert policies.admit_inbound({"type": "f"}) is False
    inner.pop()
    assert policies.admit_inbound({"type": "g"}) is False  # outer still spent
    outer.pop()
    assert policies.admit_inbound({"type": "h"}) is True

    # deny_new anywhere in the stack wins over one_more.
    outer = policies.push(dispatch_admit=policies.ONE_MORE)
    inner = policies.push(dispatch_admit=policies.DENY_NEW)
    assert policies.admit_inbound({"type": "i"}) is False
    assert outer.one_more_remaining == 1  # not consumed when denied
    inner.pop()
    outer.pop()


# ---------------------------------------------------------------------------
# Configuration remains untouched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_channel_layer_config_never_mutated():
    before = deepcopy(__import__("django").conf.settings.CHANNEL_LAYERS)

    @override_settings(CHANNEL_LAYERS=deepcopy(FRESH_LAYER_SETTING))
    async def run():
        from django.conf import settings as dj_settings

        configured = deepcopy(dj_settings.CHANNEL_LAYERS)
        backend = get_channel_layer()
        assert type(backend).__name__ == "InMemoryChannelLayer"
        capacity = backend.capacity

        holder = {}

        class C(AsyncWebsocketConsumer):
            async def connect(self):
                holder["view_type"] = type(self.channel_layer).__name__
                holder["frame"] = policies.push(
                    dispatch_admit=policies.DENY_NEW,
                    outbound_gate=policies.FORCE_DROP,
                    lifecycle_gate=policies.FORCE_CLOSE,
                )
                await self.accept()

            async def disconnect(self, code):
                holder["disconnected"] = True

        app = C()
        communicator = WebsocketCommunicator(app, "/testws/")
        connected, _ = await communicator.connect()
        assert connected
        # The consumer holds the policy view, the manager keeps the backend.
        assert holder["view_type"] == "SessionChannelLayer"
        assert type(get_channel_layer()).__name__ == "InMemoryChannelLayer"
        await communicator.send_to(text_data="ignored")
        await communicator.disconnect()
        # Backend identity/settings/capacity are unchanged.
        assert get_channel_layer() is backend
        assert get_channel_layer().capacity == capacity
        assert deepcopy(dj_settings.CHANNEL_LAYERS) == configured

    await run()
    after = deepcopy(__import__("django").conf.settings.CHANNEL_LAYERS)
    assert before == after


# ---------------------------------------------------------------------------
# Inbound axis: dispatch_admit (async websocket consumer)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inbound_deny_new_protects_inflight_then_denies():
    holder = {"received": []}

    class C(AsyncWebsocketConsumer):
        async def connect(self):
            await self.accept()

        async def receive(self, text_data=None, bytes_data=None):
            holder["received"].append(text_data)
            # Push *after* this dispatch started: the in-flight handling must
            # still complete normally; only later dispatches are constrained.
            if holder.get("frame") is None:
                holder["frame"] = policies.push(dispatch_admit=policies.DENY_NEW)
            await self.send(text_data="got:%s" % text_data)

    app = C()
    communicator = WebsocketCommunicator(app, "/testws/")
    await communicator.connect()

    await communicator.send_to(text_data="first")
    # The in-flight message was handled and answered, not cancelled or faked.
    assert await communicator.receive_from() == "got:first"

    await communicator.send_to(text_data="second")
    await communicator.send_to(text_data="third")
    assert await communicator.receive_nothing(timeout=0.2) is True
    assert holder["received"] == ["first"]
    assert holder["frame"].inbound_denied == 2
    assert holder["frame"].events[0]["type"] == "websocket.receive"
    assert holder["frame"].events[0]["reason"] == "deny_new"

    # Lifecycle teardown is still admitted, so the session closes cleanly.
    await communicator.disconnect()


@pytest.mark.asyncio
async def test_one_more_admits_exactly_one_then_denies():
    holder = {"received": []}

    class C(AsyncWebsocketConsumer):
        async def connect(self):
            holder["frame"] = policies.push(dispatch_admit=policies.ONE_MORE)
            await self.accept()

        async def receive(self, text_data=None, bytes_data=None):
            holder["received"].append(text_data)
            await self.send(text_data="got:%s" % text_data)

    app = C()
    communicator = WebsocketCommunicator(app, "/testws/")
    await communicator.connect()

    await communicator.send_to(text_data="one")
    assert await communicator.receive_from() == "got:one"
    await communicator.send_to(text_data="two")
    await communicator.send_to(text_data="three")
    assert await communicator.receive_nothing(timeout=0.2) is True
    assert holder["received"] == ["one"]
    assert holder["frame"].inbound_denied == 2
    assert holder["frame"].one_more_remaining == 0
    assert holder["frame"].events[0]["reason"] == "one_more_exhausted"
    await communicator.disconnect()


# ---------------------------------------------------------------------------
# Inbound axis: channel layer receive path on a plain AsyncConsumer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inbound_deny_new_covers_channel_layer_receive():
    with override_settings(CHANNEL_LAYERS=deepcopy(FRESH_LAYER_SETTING)):
        holder = {"layer_messages": []}

        class C(AsyncConsumer):
            async def arm(self, message):
                holder["channel_name"] = self.channel_name
                holder["frame"] = policies.push(dispatch_admit=policies.DENY_NEW)

            async def test_message(self, message):
                holder["layer_messages"].append(message)

            async def websocket_disconnect(self, message):
                raise StopConsumer()

        app = C()
        communicator = ApplicationCommunicator(app, {"type": "channel"})
        await communicator.send_input({"type": "arm"})
        await asyncio.sleep(0.1)
        channel_name = holder["channel_name"]
        frame = holder["frame"]

        layer = get_channel_layer()
        await layer.send(channel_name, {"type": "test.message", "n": 1})
        await asyncio.sleep(0.1)
        assert holder["layer_messages"] == []
        assert frame.inbound_denied == 1

        # A second layer message is rejected as well.
        await layer.send(channel_name, {"type": "test.message", "n": 2})
        await asyncio.sleep(0.1)
        assert holder["layer_messages"] == []
        assert frame.inbound_denied == 2

        # Lifecycle teardown still enters and stops the consumer.
        await communicator.send_input({"type": "websocket.disconnect"})
        await communicator.wait(timeout=1)


# ---------------------------------------------------------------------------
# Outbound axis: socket writes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_outbound_socket_deny_new_and_resume():
    holder = {}

    class C(AsyncWebsocketConsumer):
        async def connect(self):
            await self.accept()

        async def receive(self, text_data=None, bytes_data=None):
            if text_data == "mute":
                holder["frame"] = policies.push(outbound_gate=policies.DENY_NEW)
            elif text_data == "try":
                try:
                    await self.send(text_data="data")
                    holder["result"] = "ok"
                except policies.OutboundWriteDenied:
                    holder["result"] = "denied"
            elif text_data == "resume":
                await holder["frame"].apop()
                await self.send(text_data="back")

    app = C()
    communicator = WebsocketCommunicator(app, "/testws/")
    await communicator.connect()

    await communicator.send_to(text_data="mute")
    await communicator.send_to(text_data="try")
    await asyncio.sleep(0.1)
    assert holder["result"] == "denied"
    assert await communicator.receive_nothing(timeout=0.1) is True
    assert holder["frame"].outbound_denied == 1

    await communicator.send_to(text_data="resume")
    assert await communicator.receive_from() == "back"
    await communicator.disconnect()


@pytest.mark.asyncio
async def test_outbound_force_drop_fails_observably():
    holder = {}

    class C(AsyncWebsocketConsumer):
        async def connect(self):
            await self.accept()

        async def receive(self, text_data=None, bytes_data=None):
            if text_data == "drop":
                holder["frame"] = policies.push(outbound_gate=policies.FORCE_DROP)
            elif text_data == "try":
                try:
                    await self.send(text_data="data")
                    holder["result"] = "ok"
                except policies.OutboundWriteDropped:
                    holder["result"] = "dropped"
                except policies.OutboundWriteDenied:
                    holder["result"] = "denied"
            elif text_data == "resume":
                await holder["frame"].apop()
                await self.send(text_data="back")

    app = C()
    communicator = WebsocketCommunicator(app, "/testws/")
    await communicator.connect()

    await communicator.send_to(text_data="drop")
    await communicator.send_to(text_data="try")
    await asyncio.sleep(0.1)
    assert holder["result"] == "dropped"
    assert await communicator.receive_nothing(timeout=0.1) is True
    assert holder["frame"].outbound_dropped == 1

    await communicator.send_to(text_data="resume")
    assert await communicator.receive_from() == "back"
    await communicator.disconnect()


@pytest.mark.asyncio
async def test_outbound_inflight_write_not_cut_and_faked():
    holder = {}

    class C(AsyncWebsocketConsumer):
        async def connect(self):
            await self.accept()

        async def receive(self, text_data=None, bytes_data=None):
            # First write is fully in flight (and done) before the frame.
            await self.send(text_data="first")
            frame = policies.push(outbound_gate=policies.FORCE_DROP)
            try:
                await self.send(text_data="second")
            except policies.OutboundWriteDropped:
                holder["second"] = "dropped"
            await frame.apop()

    app = C()
    communicator = WebsocketCommunicator(app, "/testws/")
    await communicator.connect()

    await communicator.send_to(text_data="go")
    assert await communicator.receive_from() == "first"
    assert await communicator.receive_nothing(timeout=0.1) is True
    assert holder["second"] == "dropped"

    # The session is fully usable again afterwards.
    await communicator.send_to(text_data="ping")
    await communicator.disconnect()


@pytest.mark.asyncio
async def test_json_consumer_receive_json_and_send_json_gated():
    holder = {"received": []}

    class C(AsyncJsonWebsocketConsumer):
        async def connect(self):
            await self.accept()

        async def receive_json(self, content, **kwargs):
            holder["received"].append(content)
            cmd = content.get("cmd")
            if cmd == "mute":
                holder["frame"] = policies.push(outbound_gate=policies.DENY_NEW)
            elif cmd == "try":
                try:
                    await self.send_json({"ok": True})
                    holder["send"] = "ok"
                except policies.OutboundWriteDenied:
                    holder["send"] = "denied"
            elif cmd == "resume":
                await holder["frame"].apop()
                await self.send_json({"back": True})
            elif cmd == "block_in":
                holder["iframe"] = policies.push(dispatch_admit=policies.DENY_NEW)
            else:
                await self.send_json({"echo": content})

    app = C()
    communicator = WebsocketCommunicator(app, "/testws/")
    await communicator.connect()

    await communicator.send_json_to({"cmd": "mute"})
    await communicator.send_json_to({"cmd": "try"})
    await asyncio.sleep(0.1)
    assert holder["send"] == "denied"
    assert await communicator.receive_nothing(timeout=0.1) is True
    assert holder["frame"].outbound_denied == 1

    # Pop the outbound frame from inside the session, then push an inbound
    # frame from an admitted dispatch.
    await communicator.send_json_to({"cmd": "resume"})
    assert await communicator.receive_json_from() == {"back": True}
    await communicator.send_json_to({"cmd": "block_in"})
    await communicator.send_json_to({"cmd": "never_reaches_handler"})
    await asyncio.sleep(0.1)
    assert holder["received"][-1] == {"cmd": "block_in"}
    assert holder["iframe"].inbound_denied == 1
    await communicator.disconnect()


# ---------------------------------------------------------------------------
# Outbound axis: channel layer send / group_send
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_channel_layer_send_and_group_send_gated():
    with override_settings(CHANNEL_LAYERS=deepcopy(FRESH_LAYER_SETTING)):
        holder = {"frames": []}
        layer = get_channel_layer()
        target = await layer.new_channel()

        class C(AsyncWebsocketConsumer):
            groups = ["grp"]

            async def connect(self):
                holder["layer_view"] = type(self.channel_layer).__name__
                await self.accept()

            async def receive(self, text_data=None, bytes_data=None):
                if text_data == "layerdeny":
                    holder["frames"].append(
                        policies.push(outbound_gate=policies.DENY_NEW)
                    )
                    payload = {"type": "test.message", "n": 1}
                    holder["payload"] = payload
                    try:
                        await self.channel_layer.send(target, payload)
                        holder["send"] = "ok"
                    except policies.OutboundWriteDenied:
                        holder["send"] = "denied"
                elif text_data == "layerdrop":
                    holder["frames"].append(
                        policies.push(outbound_gate=policies.FORCE_DROP)
                    )
                    try:
                        await self.channel_layer.send(
                            target, {"type": "test.message", "n": 99}
                        )
                        holder["drop"] = "ok"
                    except policies.OutboundWriteDropped:
                        holder["drop"] = "dropped"
                elif text_data == "groupdrop":
                    holder["frames"].append(
                        policies.push(outbound_gate=policies.FORCE_DROP)
                    )
                    try:
                        await self.channel_layer.group_send(
                            "grp", {"type": "websocket.receive", "text": "injected"}
                        )
                        holder["group"] = "ok"
                    except policies.OutboundWriteDropped:
                        holder["group"] = "dropped"
                elif text_data == "popall":
                    while holder["frames"]:
                        await holder["frames"].pop().apop()
                else:
                    await self.send(text_data=text_data)

        app = C()
        communicator = WebsocketCommunicator(app, "/testws/")
        await communicator.connect()
        assert holder["layer_view"] == "SessionChannelLayer"

        await communicator.send_to(text_data="layerdeny")
        await asyncio.sleep(0.1)
        assert holder["send"] == "denied"
        # Nothing was enqueued by the denied write.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(layer.receive(target), 0.2)
        # Caller-held buffer is intact and was not faked as delivered.
        assert holder["payload"] == {"type": "test.message", "n": 1}

        await communicator.send_to(text_data="layerdrop")
        await asyncio.sleep(0.1)
        assert holder["drop"] == "dropped"
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(layer.receive(target), 0.2)

        await communicator.send_to(text_data="groupdrop")
        await asyncio.sleep(0.1)
        assert holder["group"] == "dropped"
        assert await communicator.receive_nothing(timeout=0.1) is True

        await communicator.send_to(text_data="popall")
        await asyncio.sleep(0.1)

        # Existing layer behaviour (including its own errors) is aligned again.
        await layer.send(target, {"type": "test.message", "n": 2})
        message = await asyncio.wait_for(layer.receive(target), 0.5)
        assert message == {"type": "test.message", "n": 2}
        await layer.group_send("grp", {"type": "websocket.receive", "text": "after"})
        assert await communicator.receive_from() == "after"

        await communicator.disconnect()


# ---------------------------------------------------------------------------
# Lifecycle axis: deny_accept
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifecycle_deny_accept_is_observable_and_recoverable():
    holder = {}

    class C(AsyncWebsocketConsumer):
        async def connect(self):
            frame = policies.push(lifecycle_gate=policies.DENY_ACCEPT)
            holder["frame"] = frame
            try:
                await self.accept()
            except policies.AcceptDenied:
                holder["denied"] = True
            await frame.apop()
            # Once the frame is gone the accept succeeds.
            await self.accept()

        async def receive(self, text_data=None, bytes_data=None):
            await self.send(text_data="pong")

    app = C()
    communicator = WebsocketCommunicator(app, "/testws/")
    connected, _ = await communicator.connect()
    assert connected is True
    assert holder["denied"] is True
    assert holder["frame"].accept_denied == 1
    # Exactly one accept reached the wire, and it was not silently marked.
    assert await communicator.receive_nothing(timeout=0.1) is True
    await communicator.send_to(text_data="ping")
    assert await communicator.receive_from() == "pong"
    await communicator.disconnect()


# ---------------------------------------------------------------------------
# Lifecycle axis: force_close on pop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_force_close_closes_session_accepted_inside_frame():
    holder = {}

    class C(AsyncWebsocketConsumer):
        async def connect(self):
            holder["frame"] = policies.push(lifecycle_gate=policies.FORCE_CLOSE)
            await self.accept()

        async def receive(self, text_data=None, bytes_data=None):
            if text_data == "pop":
                await holder["frame"].apop()

    app = C()
    communicator = WebsocketCommunicator(app, "/testws/")
    connected, _ = await communicator.connect()
    assert connected is True
    assert await communicator.receive_nothing(timeout=0.1) is True

    await communicator.send_to(text_data="pop")
    message = await communicator.receive_output()
    assert message["type"] == "websocket.close"
    assert message["code"] == policies.FORCE_CLOSE_CODE
    assert holder["frame"].force_closed == 1
    await communicator.disconnect()


@pytest.mark.asyncio
async def test_force_close_does_not_touch_session_accepted_before_frame():
    holder = {}

    class C(AsyncWebsocketConsumer):
        async def connect(self):
            await self.accept()
            holder["frame"] = policies.push(lifecycle_gate=policies.FORCE_CLOSE)

        async def receive(self, text_data=None, bytes_data=None):
            if text_data == "pop":
                await holder["frame"].apop()
                await self.send(text_data="alive")

    app = C()
    communicator = WebsocketCommunicator(app, "/testws/")
    await communicator.connect()
    await communicator.send_to(text_data="pop")
    # The session keeps working; no close was emitted by the pop.
    assert await communicator.receive_from() == "alive"
    assert await communicator.receive_nothing(timeout=0.1) is True
    assert holder["frame"].force_closed == 0
    await communicator.disconnect()


@pytest.mark.asyncio
async def test_force_close_bubbles_accept_to_outer_frame():
    holder = {}

    class C(AsyncWebsocketConsumer):
        async def connect(self):
            holder["outer"] = policies.push(lifecycle_gate=policies.FORCE_CLOSE)
            holder["inner"] = policies.push()  # honor_cfg inner frame
            await self.accept()
            holder["inner"].pop()  # ownership bubbles to outer, no close yet

        async def receive(self, text_data=None, bytes_data=None):
            if text_data == "pop":
                await holder["outer"].apop()

    app = C()
    communicator = WebsocketCommunicator(app, "/testws/")
    await communicator.connect()
    assert await communicator.receive_nothing(timeout=0.1) is True
    assert holder["inner"].force_closed == 0

    await communicator.send_to(text_data="pop")
    message = await communicator.receive_output()
    assert message["type"] == "websocket.close"
    assert holder["outer"].force_closed == 1
    assert holder["inner"].force_closed == 0
    await communicator.disconnect()


@pytest.mark.asyncio
async def test_force_close_runs_on_exception_unwind():
    class C(AsyncWebsocketConsumer):
        async def connect(self):
            await self.accept()

        async def receive(self, text_data=None, bytes_data=None):
            async with policies.push(lifecycle_gate=policies.FORCE_CLOSE):
                # Accepted while the frame is active (the harness queues the
                # duplicate accept; real servers see the close right after).
                await self.accept()
                if text_data == "boom":
                    raise RuntimeError("boom")

    app = C()
    communicator = WebsocketCommunicator(app, "/testws/")
    await communicator.connect()
    await communicator.send_to(text_data="boom")

    accept = await communicator.receive_output()
    assert accept["type"] == "websocket.accept"
    close = await communicator.receive_output()
    assert close["type"] == "websocket.close"
    # The original exception still propagates; the frame popped cleanly.
    with pytest.raises(RuntimeError, match="boom"):
        await communicator.wait(timeout=1)


@pytest.mark.asyncio
async def test_force_close_neighbor_session_not_closed():
    holder_a = {}
    holder_b = {"views": []}

    class A(AsyncWebsocketConsumer):
        async def connect(self):
            holder_a["frame"] = policies.push(lifecycle_gate=policies.FORCE_CLOSE)
            await self.accept()

        async def receive(self, text_data=None, bytes_data=None):
            if text_data == "pop":
                await holder_a["frame"].apop()

    class B(AsyncWebsocketConsumer):
        async def connect(self):
            await self.accept()

        async def receive(self, text_data=None, bytes_data=None):
            holder_b["views"].append(policies.current_view())
            await self.send(text_data="b:%s" % text_data)

    app_a = A()
    app_b = B()
    comm_a = WebsocketCommunicator(app_a, "/a/")
    comm_b = WebsocketCommunicator(app_b, "/b/")
    await comm_a.connect()
    await comm_b.connect()

    # B serves normally while A has a frame pushed.
    await comm_b.send_to(text_data="ping")
    assert await comm_b.receive_from() == "b:ping"
    assert holder_b["views"][-1].depth == 0

    # Popping A's frame closes only A.
    await comm_a.send_to(text_data="pop")
    close = await comm_a.receive_output()
    assert close["type"] == "websocket.close"

    await comm_b.send_to(text_data="still-here")
    assert await comm_b.receive_from() == "b:still-here"
    assert await comm_b.receive_nothing(timeout=0.1) is True

    await comm_a.disconnect()
    await comm_b.disconnect()


# ---------------------------------------------------------------------------
# Cross-consumer scope isolation on one event loop (inbound + outbound)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_consumers_have_independent_stacks():
    holder_a = {"received": []}
    holder_b = {"received": [], "views": []}

    class A(AsyncWebsocketConsumer):
        async def connect(self):
            holder_a["frame"] = policies.push(
                dispatch_admit=policies.DENY_NEW,
                outbound_gate=policies.DENY_NEW,
            )
            await self.accept()

        async def receive(self, text_data=None, bytes_data=None):
            holder_a["received"].append(text_data)

    class B(AsyncWebsocketConsumer):
        async def connect(self):
            await self.accept()

        async def receive(self, text_data=None, bytes_data=None):
            holder_b["received"].append(text_data)
            holder_b["views"].append(policies.current_view())
            await self.send(text_data="b:%s" % text_data)

    comm_a = WebsocketCommunicator(A(), "/a/")
    comm_b = WebsocketCommunicator(B(), "/b/")
    await comm_a.connect()
    await comm_b.connect()

    # A rejects new inbound dispatches...
    await comm_a.send_to(text_data="a1")
    assert await comm_a.receive_nothing(timeout=0.2) is True
    assert holder_a["received"] == []
    assert holder_a["frame"].inbound_denied == 1

    # ...while B, on the same event loop with no frame pushed, is untouched
    # for both inbound admission and outbound writes.
    await comm_b.send_to(text_data="b1")
    assert await comm_b.receive_from() == "b:b1"
    assert holder_b["received"] == ["b1"]
    assert holder_b["views"][-1].depth == 0
    assert holder_b["views"][-1].inbound_open is True

    # Frames cannot cross: B never accumulated denial events.
    assert holder_a["frame"].outbound_denied == 0

    await comm_a.disconnect()
    await comm_b.disconnect()


# ---------------------------------------------------------------------------
# Nested outbound frames merge; inner pop restores outer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nested_outbound_frames_merge_and_restore():
    holder = {"results": []}

    class C(AsyncWebsocketConsumer):
        async def connect(self):
            await self.accept()

        async def _attempt(self):
            try:
                await self.send(text_data="x")
                holder["results"].append("ok")
            except policies.OutboundWriteDropped:
                holder["results"].append("dropped")
            except policies.OutboundWriteDenied:
                holder["results"].append("denied")

        async def receive(self, text_data=None, bytes_data=None):
            if text_data == "nested":
                outer = policies.push(outbound_gate=policies.DENY_NEW)
                inner = policies.push(outbound_gate=policies.FORCE_DROP)
                view_inside = policies.current_view()
                await self._attempt()  # force_drop wins across the stack
                await inner.apop()
                view_after_inner = policies.current_view()
                await self._attempt()  # outer deny applies again
                await outer.apop()
                view_after_outer = policies.current_view()
                holder["views"] = (view_inside, view_after_inner, view_after_outer)
            elif text_data == "ping":
                await self.send(text_data="pong")

    app = C()
    communicator = WebsocketCommunicator(app, "/testws/")
    await communicator.connect()

    await communicator.send_to(text_data="nested")
    await asyncio.sleep(0.1)
    assert holder["results"] == ["dropped", "denied"]
    assert await communicator.receive_nothing(timeout=0.1) is True
    inside, after_inner, after_outer = holder["views"]
    assert inside.depth == 2
    assert inside.outbound_gate == policies.FORCE_DROP
    assert after_inner.depth == 1
    assert after_inner.outbound_gate == policies.DENY_NEW
    assert after_outer.depth == 0
    assert after_outer.outbound_gate == policies.HONOR_CFG

    await communicator.send_to(text_data="ping")
    assert await communicator.receive_from() == "pong"
    await communicator.disconnect()


# ---------------------------------------------------------------------------
# Sync consumers (context propagates across the asgiref bridge)
# ---------------------------------------------------------------------------


def _make_sync_classes(holder):
    class Inbound(WebsocketConsumer):
        def connect(self):
            if holder.get("mode") == "one_more":
                holder["frame"] = policies.push(dispatch_admit=policies.ONE_MORE)
            elif holder.get("mode") == "force_close":
                holder["frame"] = policies.push(lifecycle_gate=policies.FORCE_CLOSE)
            elif holder.get("mode") == "deny_accept":
                frame = policies.push(lifecycle_gate=policies.DENY_ACCEPT)
                try:
                    self.accept()
                except policies.AcceptDenied:
                    holder["accept_denied"] = True
                frame.pop()
            self.accept()

        def receive(self, text_data=None, bytes_data=None):
            holder["received"].append(text_data)
            if text_data == "arm" and holder.get("deny_frame") is None:
                holder["deny_frame"] = policies.push(
                    dispatch_admit=policies.DENY_NEW
                )
                self.send(text_data="got:%s" % text_data)
            elif text_data == "mute":
                holder["out_frame"] = policies.push(
                    outbound_gate=policies.DENY_NEW
                )
            elif text_data == "try":
                try:
                    self.send(text_data="x")
                    holder["send_result"] = "ok"
                except policies.OutboundWriteDenied:
                    holder["send_result"] = "denied"
            elif text_data == "resume":
                holder["out_frame"].pop()
                self.send(text_data="back")
            elif text_data == "force_pop":
                holder["frame"].pop()
            else:
                self.send(text_data="got:%s" % text_data)

    return Inbound


@pytest.mark.asyncio
async def test_sync_consumer_inbound_outbound_lifecycle_gates():
    holder = {"received": []}
    consumer_cls = _make_sync_classes(holder)

    # 1) In-flight message completes; later ones are denied.
    communicator = WebsocketCommunicator(consumer_cls(), "/testws/")
    await communicator.connect()
    await communicator.send_to(text_data="arm")
    assert await communicator.receive_from() == "got:arm"
    await communicator.send_to(text_data="next")
    assert await communicator.receive_nothing(timeout=0.2) is True
    assert holder["received"] == ["arm"]
    assert holder["deny_frame"].inbound_denied == 1
    await communicator.disconnect()

    # 2) one_more admits exactly one.
    holder["mode"] = "one_more"
    holder["received"] = []
    communicator = WebsocketCommunicator(consumer_cls(), "/testws/")
    await communicator.connect()
    await communicator.send_to(text_data="one")
    assert await communicator.receive_from() == "got:one"
    await communicator.send_to(text_data="two")
    assert await communicator.receive_nothing(timeout=0.2) is True
    assert holder["received"] == ["one"]
    assert holder["frame"].one_more_remaining == 0
    await communicator.disconnect()

    # 3) Outbound deny then resume.
    holder["mode"] = None
    holder["received"] = []
    communicator = WebsocketCommunicator(consumer_cls(), "/testws/")
    await communicator.connect()
    await communicator.send_to(text_data="mute")
    await communicator.send_to(text_data="try")
    await asyncio.sleep(0.1)
    assert holder["send_result"] == "denied"
    assert await communicator.receive_nothing(timeout=0.1) is True
    await communicator.send_to(text_data="resume")
    assert await communicator.receive_from() == "back"
    await communicator.disconnect()

    # 4) force_close pop emits an observable close.
    holder["mode"] = "force_close"
    communicator = WebsocketCommunicator(consumer_cls(), "/testws/")
    await communicator.connect()
    await communicator.send_to(text_data="force_pop")
    message = await communicator.receive_output()
    assert message["type"] == "websocket.close"
    assert message["code"] == policies.FORCE_CLOSE_CODE
    assert holder["frame"].force_closed == 1
    await communicator.disconnect()

    # 5) deny_accept rejects once, then the accept succeeds.
    holder["mode"] = "deny_accept"
    communicator = WebsocketCommunicator(consumer_cls(), "/testws/")
    connected, _ = await communicator.connect()
    assert connected is True
    assert holder["accept_denied"] is True
    await communicator.disconnect()


@pytest.mark.asyncio
async def test_sync_json_consumer_outbound_gate():
    holder = {}

    class C(JsonWebsocketConsumer):
        def connect(self):
            self.accept()

        def receive_json(self, content, **kwargs):
            if content.get("cmd") == "drop":
                holder["frame"] = policies.push(outbound_gate=policies.FORCE_DROP)
            elif content.get("cmd") == "try":
                try:
                    self.send_json({"ok": True})
                    holder["result"] = "ok"
                except policies.OutboundWriteDropped:
                    holder["result"] = "dropped"

    communicator = WebsocketCommunicator(C(), "/testws/")
    await communicator.connect()
    await communicator.send_json_to({"cmd": "drop"})
    await communicator.send_json_to({"cmd": "try"})
    await asyncio.sleep(0.1)
    assert holder["result"] == "dropped"
    assert holder["frame"].outbound_dropped == 1
    assert await communicator.receive_nothing(timeout=0.1) is True
    await communicator.disconnect()
