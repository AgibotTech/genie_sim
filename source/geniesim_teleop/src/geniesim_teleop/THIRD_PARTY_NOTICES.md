# Third-party notices

The Genie Sim teleoperation code is licensed under MPL-2.0. Third-party
dependencies retain their own licenses and copyright notices.

## Wuji SDK

- **Project:** Wuji SDK (`wuji-sdk`, imported as `wuji_sdk`)
- **Upstream:** https://github.com/wuji-technology/wuji-sdk
- **License:** MIT
- **Copyright:** Copyright (c) 2026 Wuji Technology
- **Upstream license:** https://github.com/wuji-technology/wuji-sdk/blob/main/LICENSE
- **Local license copy:** [LICENSE.WUJI_SDK](THIRD_PARTY_LICENSES/LICENSE.WUJI_SDK)
- **Usage:** `wuji/hand_bridge.py` accesses the SDK for glove capture and hand
  retargeting; `wuji/sdk.py` reuses that adapter for the combined Pico/glove workflow.

The license text is reproduced verbatim from the upstream repository. This
notice and the license copy are included in the `geniesim_teleop` package.
