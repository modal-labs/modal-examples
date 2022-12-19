from text_to_pokemon import pokemon_naming


def test_prompt_2_name_basic_matching():
    test_candidates = {
        "sleepmon",
        "bulbasaur",
        "bulbasaur",
        "foobar",
    }
    assert (
        pokemon_naming.prompt_2_name(
            prompt="sleepy monkey",
            candidates=test_candidates,
        )
        == "sleepmon"
    )
    assert (
        pokemon_naming.prompt_2_name(
            prompt="monkey asleep",
            candidates=test_candidates,
        )
        == "sleepmon"
    )
    # TODO(erikbern): reenable this. See #151 also.
    # assert (
    #     pokemon_naming.prompt_2_name(
    #         prompt="garlic bulb",
    #         candidates=test_candidates,
    #     )
    #     == "bulbasaur"
    # )
    assert (
        pokemon_naming.prompt_2_name(
            prompt="f",
            candidates=test_candidates,
        )
        == "foobar"
    )
