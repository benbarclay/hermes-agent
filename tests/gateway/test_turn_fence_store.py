"""The owner's drain fences each claimed turn's writes on the store that issued the claim.

Real GatewayRunner, DiscordAdapter, SessionAuthority and drain over a real SessionDB; only the
agent's conversation is a stand-in. Inside the turn, the authority's store sees the turn as its
own (``hermes_state_fence.store_turn``) with the claim's fence, and a write made after another
writer moved the session's generation is refused without landing.
"""
import contextvars

import pytest

from tests.gateway.test_messaging_stop_owner_turn import _owner, _until


@pytest.mark.asyncio
async def test_a_drained_turn_is_fenced_on_its_authority_store(monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from hermes_scope import Fence
    from hermes_state_errors import TurnFenceSupersededError
    from hermes_state_fence import store_turn
    from hermes_state_runtime import list_session_admissions

    runner, adapter, _sent = _owner(monkeypatch)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    key = runner.session_store._generate_session_key(source)
    db = authority.db
    seen = {}

    def bump(sid):
        db._write_sql('UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?', (sid,))

    async def turn(event, source, quick_key, run_generation):
        mark = store_turn(db)
        seen['mark'] = mark
        db.append_message(mark.session, 'user', 'held')
        contextvars.Context().run(bump, mark.session)  # another writer supersedes the claim
        try:
            db.append_message(mark.session, 'assistant', 'stale')
        except TurnFenceSupersededError:
            seen['refused'] = True
        execution_result.get()['result'] = {'final_response': 'done', 'messages': []}
        return 'done'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    await adapter.handle_message(MessageEvent(text='A', source=source, message_id='A'))
    await _until(lambda: 'mark' in seen)
    owner = runner.session_store.peek_session_id(key)
    await _until(lambda: [r['status'] for r in list_session_admissions(
        db, session_id=owner, pending_only=False)] not in (['queued'], ['started']))
    [row] = list_session_admissions(db, session_id=owner, pending_only=False)
    mark = seen['mark']
    assert mark is not None, 'the drained turn was not fenced on its authority store'
    assert (mark.session, mark.turn, mark.fence) == (owner, row['admission_id'],
                                                     Fence(authority.epoch, row['generation']))
    assert seen.get('refused') is True
    assert [m['content'] for m in db.get_messages(owner)] == ['held']
    assert not mark.live  # the mark closed with the turn's body
