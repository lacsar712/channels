import functools

from asgiref.sync import async_to_sync

from . import DEFAULT_CHANNEL_LAYER, policies
from .db import aclose_old_connections, database_sync_to_async
from .exceptions import StopConsumer
from .layers import get_channel_layer
from .utils import await_many_dispatch


def get_handler_name(message):
    """
    Looks at a message, checks it has a sensible type, and returns the
    handler name for that type.
    """
    # Check message looks OK
    if "type" not in message:
        raise ValueError("Incoming message has no 'type' attribute")
    # Extract type and replace . with _
    handler_name = message["type"].replace(".", "_")
    if handler_name.startswith("_"):
        raise ValueError("Malformed type in message (leading underscore)")
    return handler_name


class AsyncConsumer:
    """
    Base consumer class. Implements the ASGI application spec, and adds on
    channel layer management and routing of events to named methods based
    on their type.
    """

    _sync = False
    channel_layer_alias = DEFAULT_CHANNEL_LAYER

    async def __call__(self, scope, receive, send):
        """
        Dispatches incoming messages to type-based handlers asynchronously.
        """
        self.scope = scope

        # Initialize channel layer. The consumer holds a policy-aware view
        # for the duration of this session; the backend and its configuration
        # are left untouched.
        self.channel_layer = policies.gate_channel_layer(
            get_channel_layer(self.channel_layer_alias)
        )
        if self.channel_layer is not None:
            self.channel_name = await self.channel_layer.new_channel()
            self.channel_receive = functools.partial(
                self.channel_layer.receive, self.channel_name
            )
        # Store send function
        if self._sync:
            self.base_send = async_to_sync(send)
        else:
            self.base_send = send
        # Every consumer session gets a fresh, isolated runtime policy stack
        # and policy-owned session state. The reset tokens restore any
        # enclosing scope on exit.
        if self._sync:
            session_token = policies.bind_session(sync_close=self._policy_force_close)
        else:
            session_token = policies.bind_session(async_close=self._policy_force_close)
        stack_token = policies.reset_stack()
        # Pass messages in from channel layer or client to dispatch method
        try:
            if self.channel_layer is not None:
                await await_many_dispatch(
                    [receive, self.channel_receive], self._policy_dispatch
                )
            else:
                await await_many_dispatch([receive], self._policy_dispatch)
        except StopConsumer:
            # Exit cleanly
            pass
        finally:
            policies.restore_stack(stack_token)
            policies.end_session(session_token)

    async def _policy_dispatch(self, message):
        """
        Session-policy admission boundary for inbound messages. Every
        inbound message (from the client receive callable or from the channel
        layer) passes through here before reaching ``dispatch`` and therefore
        any consumer handler, including WebsocketConsumer.receive /
        receive_json. A rejected message is dropped observably (see the frame
        event log) without tearing down the consumer.
        """
        if policies.admit_inbound(message):
            await self.dispatch(message)

    async def _policy_force_close(self, code=1000):
        """
        Emit a raw lifecycle close when a force_close frame pops. This never
        goes through the outbound data gate.
        """
        await self.base_send({"type": "websocket.close", "code": code})

    async def dispatch(self, message):
        """
        Works out what to do with a message.
        """
        handler = getattr(self, get_handler_name(message), None)
        if handler:
            await aclose_old_connections()
            await handler(message)
        else:
            raise ValueError("No handler for message type %s" % message["type"])

    async def send(self, message):
        """
        Overrideable/callable-by-subclasses send method.

        Outbound ASGI messages consult the session outbound gate, while the
        websocket accept/close lifecycle messages consult the lifecycle gate.
        """
        message_type = message.get("type")
        if message_type == "websocket.accept":
            policies.guard_accept()
            await self.base_send(message)
            policies.note_accepted()
        elif message_type == "websocket.close":
            policies.note_close_requested()
            await self.base_send(message)
        else:
            policies.guard_outbound({"via": "asgi.send", "type": message_type})
            await self.base_send(message)

    @classmethod
    def as_asgi(cls, **initkwargs):
        """
        Return an ASGI v3 single callable that instantiates a consumer instance
        per scope. Similar in purpose to Django's as_view().

        initkwargs will be used to instantiate the consumer instance.
        """

        async def app(scope, receive, send):
            consumer = cls(**initkwargs)
            return await consumer(scope, receive, send)

        app.consumer_class = cls
        app.consumer_initkwargs = initkwargs

        # take name and docstring from class
        functools.update_wrapper(app, cls, updated=())
        return app


class SyncConsumer(AsyncConsumer):
    """
    Synchronous version of the consumer, which is what we write most of the
    generic consumers against (for now). Calls handlers in a threadpool and
    uses CallBouncer to get the send method out to the main event loop.

    It would have been possible to have "mixed" consumers and auto-detect
    if a handler was awaitable or not, but that would have made the API
    for user-called methods very confusing as there'd be two types of each.
    """

    _sync = True

    @database_sync_to_async
    def dispatch(self, message):
        """
        Dispatches incoming messages to type-based handlers asynchronously.
        """
        # Get and execute the handler
        handler = getattr(self, get_handler_name(message), None)
        if handler:
            handler(message)
        else:
            raise ValueError("No handler for message type %s" % message["type"])

    def _policy_force_close(self, code=1000):
        self.base_send({"type": "websocket.close", "code": code})

    def send(self, message):
        """
        Overrideable/callable-by-subclasses send method.

        Mirrors the async version; context variables propagate across the
        asgiref sync/async bridge, so the live session stack is read here.
        """
        message_type = message.get("type")
        if message_type == "websocket.accept":
            policies.guard_accept()
            self.base_send(message)
            policies.note_accepted()
        elif message_type == "websocket.close":
            policies.note_close_requested()
            self.base_send(message)
        else:
            policies.guard_outbound({"via": "asgi.send", "type": message_type})
            self.base_send(message)
