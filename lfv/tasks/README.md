# Tasks

Task definitions live here. A task should describe the semantic manipulation
problem independently from a single dataset path or model implementation.

Examples:

- `hand_pouring`: grasp a cup and pour into a bowl.
- future tasks: drawer opening, sweeping, placing, pressing, stirring.

Expected task-level metadata:

- manipulated object name and prompts;
- reference or target object name and prompts;
- coordinate frame convention;
- task stages such as contact, grasp, transfer, and interaction;
- available labels, such as contact heat, grasp pseudo labels, and SE(3)
  object trajectories.

Keep task definitions lightweight. Data loading, model implementation, and
training logic should live in their own packages.

