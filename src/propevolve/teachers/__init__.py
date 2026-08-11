"""Training-only teacher interface and authenticated adapters."""

from .base import BaseTeacher
from .composition import (
    CombinedTeacherTargets,
    TeacherTargetSource,
    load_teacher_targets,
)
from .expansion import ExpansionTeacher
from .regime import RegimeTeacher

__all__ = [
    "BaseTeacher",
    "CombinedTeacherTargets",
    "ExpansionTeacher",
    "RegimeTeacher",
    "TeacherTargetSource",
    "load_teacher_targets",
]
