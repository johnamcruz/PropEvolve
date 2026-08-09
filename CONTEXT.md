# Domain context

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
