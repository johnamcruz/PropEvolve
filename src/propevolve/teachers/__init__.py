"""Training-only teacher interface and authenticated adapters."""

from .base import BaseTeacher
from .composition import (
    agent_teacher_settings,
    CombinedTeacherTargets,
    TeacherTargetSource,
    load_teacher_targets,
)
from .expansion import ExpansionTeacher
from .regime import RegimeTeacher
from .trend import TrendTeacher

__all__ = [
    "agent_teacher_settings",
    "BaseTeacher",
    "CombinedTeacherTargets",
    "ExpansionTeacher",
    "RegimeTeacher",
    "TrendTeacher",
    "TeacherTargetSource",
    "load_teacher_targets",
]
