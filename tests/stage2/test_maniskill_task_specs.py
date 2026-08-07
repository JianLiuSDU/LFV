from lfv_sim.maniskill.specs import get_task_spec


def test_picknplace_task_uses_banana_and_plate_roles():
    spec = get_task_spec("picknplace")

    assert spec.env_id == "LFVPickBananaPlate-v1"
    assert spec.manipulated_query == "banana"
    assert spec.target_query == "plate"
    assert spec.manipulated_entity_attr == "banana"
    assert spec.reference_entity_attr == "plate"
