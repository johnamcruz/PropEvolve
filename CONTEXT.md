# Domain context

- **Entry decision**: ENTER means taking either a Long or Short trade; WAIT
  means sitting out. Its score boundary is `max(Long, Short) > WAIT`.
- **Direction decision**: when the economic label calls for ENTER, evaluate
  LONG versus SHORT separately. Entering on the wrong side fails direction and
  the complete trade action, but does not also fail the ENTER decision.
- **Management decision**: an open position learns HOLD versus CLOSE only.
- **Partial mastery**: retain each correct applicable boundary and correct each
  incorrect one independently. WAIT targets have no direction supervision.
- **Trading evidence**: future target-before-stop and excursion outcomes are
  labels, never policy inputs. Teacher-free policy evaluation must use the same
  decision definitions as training, retention, and checkpoint selection.

- **Challenge P&L**: net realized or marked-to-market profit relative to the
  challenge starting balance; the model does not depend on the broker's
  absolute balance coordinate.
- **MLL floor**: the challenge P&L level at or below which the account blows.
- **EOD trail**: at the 5:00 p.m. America/Chicago session boundary, the MLL
  floor ratchets to at most one maximum-loss allowance below realized P&L and
  never moves backward.
- **Passmark lock**: once realized P&L reaches one maximum-loss allowance, the
  MLL floor locks permanently at zero challenge P&L.
- **Pass**: net challenge P&L reaches the configured profit target.
- **Blow**: intrabar net equity touches or crosses the effective MLL floor;
  blow has priority over a recovering bar close.
- **Timeout**: the episode window ends without a pass or blow; any open
  position is liquidated with costs.
- **Golden trajectory**: an immutable input/action sequence and expected
  economic receipt captured from the trusted simulator contract. PropEvolve
  must reproduce it through its public `reset`/`step` interface.
