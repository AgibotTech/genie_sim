# Robot description sources and license status

Agent2Act uses these files for local forward/inverse kinematics and joint limits.
It does not load their visual meshes, collision geometry, or inertial properties.
The full license inventory is in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
[provenance.json](provenance.json) records source revisions, hashes, and joint
fingerprints for both files.

| File | Source evidence | Redistribution status |
|---|---|---|
| `G2_NO_GRIPPER.urdf` | Exact public Genie Sim benchmark copy | MPL-2.0; license text included |
| `G2_omnipicker_fixed_dual.urdf` | Exact public Genie Sim data-collection copy | MPL-2.0; license text included |

Both files are unchanged byte-identical copies of the upstream sources.

No mesh assets are included in this directory. A path to an external mesh does not
license that mesh. Do not replace these URDFs with a similarly named public model:
different G2 versions have different joint coordinates and would change FK/IK.

## Distribution

The source distribution includes these two URDFs, this inventory, and license texts.
The Docker image copies the examples and their notices. The Python wheel contains
the runtime only and does not bundle the URDF examples. Packaging success is not a
license clearance; the inventory, hashes, and license evidence are recorded in
`provenance.json` and the third-party notices linked above.
