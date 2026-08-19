"""Synthetic radar objects and analytic fields for tests and examples.

Validating an interpolation operator needs a ground truth the operator did
not see.  The generators here build Py-ART ``Radar`` objects on realistic
scan geometries, painted with analytic fields whose exact value at any
(time, position) is known -- for example a scene advected rigidly at a
prescribed velocity, so the true intermediate volume is available in closed
form.

Status
------
Scaffolding only: the implementation lands in follow-up pull requests.
"""

from __future__ import annotations

__all__: list[str] = []
