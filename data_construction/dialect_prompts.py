"""Few-shot translation prompts, shared by the two dialect translation scripts.

Every dialect uses the same prompt shape and differs only in its display name and its
three example sentences, so both live in one table rather than one function each. The
examples follow the EnDive framework (https://arxiv.org/abs/2504.07100).

The rendered strings are byte-for-byte what generated the released dataset — changing
the wording or the examples would change the data, so treat this table as fixed.
"""

# dialect key -> (display name used verbatim in the prompts, three example sentences)
DIALECT_SPECS = {
    "ChcE": ("Chicano English (ChcE)", [
        "When people wanna fight me I'm like \"well okay, well then I'll fight you.\"",
        "They were saying that they had a lot of problems at Garner because it was a lot of fights and stuff.",
        "I ain't really thinking about getting with J. or any other guy",
    ]),
    "CollSgE": ("Colloquial Singapore English (Singlish) (CollSgE)", [
        "But after a while it become quite senseless to me.",
        "And got to know this kind-hearted scholar who shelter her with Ø umbrella when it was raining.",
        "The cake John buy one always very nice to eat.",
    ]),
    "AAVE": ("African American Vernacular English (AAVE)", [
        "I was bewildered, but I knew dat it was no gud asking his ass to explain.",
        "Cochran pontificated windily for da camera.",
        "I don’t want them to follow in my footsteps, as I ain’t go to no college, but I want them to go.",
    ]),
    "IndE": ("Indian English (IndE)", [
        "It was not too much common. Getting the accommodation has become very much difficult.",
        "During monsoon we get lot of rain and then gets very soggy and sultry.",
        "This is the second time that such an object had been sighted here.",
    ]),
    "JamE": ("Jamaican English (JamE)", [
        "Hill had initially been indicted with the Canute and the Michelle Saddler and their three companies.",
        "The autopsy performed on Mae's torso shortly after it was found, revealed that her body was cut into pieces by a power machine saw.",
        "The culture of the region has been unique in combining British and Western influences with African and Asian lifestyles.",
    ]),
}


def system_prompt(dialect: str) -> str:
    name, _ = DIALECT_SPECS[dialect]
    return f"You are a language model capable of translating text into {name}."


def few_shot_prompt(dialect: str, text: str) -> str:
    name, examples = DIALECT_SPECS[dialect]
    numbered = "".join(f"{i}. {e}\n" for i, e in enumerate(examples, 1))
    return (f"Here are examples of {name}:\n"
            f"{numbered}"
            f"\nHere is the input text: {text}\n"
            f"Please rewrite the input text in {name}.")
