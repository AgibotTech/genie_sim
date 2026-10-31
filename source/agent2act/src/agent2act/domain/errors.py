# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Errors crossing application boundaries."""


class ContractError(ValueError):
    """Invalid observation, decision, configuration or generated action."""


class ConfigurationError(ValueError):
    """Invalid deployment configuration or registration."""


class PlanningExhaustedError(ContractError):
    """No valid action could be produced within the replanning budget."""


class LocalIKConvergenceError(ContractError):
    """A bounded local IK search failed; the requested motion was not issued."""


class ExpiredProposalError(ContractError):
    """Review exceeded the candidate lifetime; replanning needs a new observation."""


class TransientBackendError(RuntimeError):
    """A temporary backend failure exhausted its bounded request retry budget."""


class ModelIdentityError(RuntimeError):
    """The configured model could not be verified within the request budget."""
