"""Builders for tiny hand-written NXT corpora used by the Task 1 tests."""

from pathlib import Path

NITE_NAMESPACE = "http://nite.sourceforge.net/"
MEETING = "Mtg001"


def word(
    word_id: str,
    text: str = "",
    start: str | None = None,
    end: str | None = None,
    kind: str = "w",
    description: str | None = None,
) -> str:
    attributes = f'nite:id="{word_id}"'
    if start is not None:
        attributes += f' starttime="{start}"'
    if end is not None:
        attributes += f' endtime="{end}"'
    if description is not None:
        attributes += f' description="{description}"'
    return f"<{kind} {attributes}>{text}</{kind}>"


def words_href(first: str, last: str | None = None, agent: str = "A") -> str:
    target = f"id({first})" if last is None else f"id({first})..id({last})"
    return f"{MEETING}.{agent}.words.xml#{target}"


def act(
    act_id: str,
    start: str | None,
    end: str | None,
    type_: str | None = "s",
    *,
    original_type: str | None = None,
    channel: str = "c1",
    participant: str = "me001",
    adjacency: str | None = None,
    comment: str | None = None,
    href: str | None = None,
) -> str:
    attributes = [f'nite:id="{act_id}"']
    optional = (
        ("starttime", start),
        ("endtime", end),
        ("type", type_),
        ("original-type", original_type),
        ("channel", channel),
        ("participant", participant),
        ("adjacency", adjacency),
        ("comment", comment),
    )
    for name, value in optional:
        if value is not None:
            attributes.append(f'{name}="{value}"')
    child = f'<nite:child href="{href}"/>' if href is not None else ""
    return f"<dialogueact {' '.join(attributes)}>{child}</dialogueact>"


def _write_xml(path: Path, root_id: str, elements: list[str]) -> None:
    body = "\n".join(elements)
    text = (
        '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
        f'<nite:root xmlns:nite="{NITE_NAMESPACE}" nite:id="{root_id}">\n{body}\n</nite:root>\n'
    )
    path.write_bytes(text.encode("iso-8859-1"))


def write_corpus(
    root: Path,
    *,
    words: dict[str, list[str]],
    acts: dict[str, list[str]],
    transcripts: list[str] | None = None,
    meeting: str = MEETING,
) -> Path:
    for folder in ("DialogueActs", "Words", "transcripts"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for agent, elements in words.items():
        path = root / "Words" / f"{meeting}.{agent}.words.xml"
        _write_xml(path, f"{meeting}.{agent}.words", elements)
    for agent, elements in acts.items():
        path = root / "DialogueActs" / f"{meeting}.{agent}.dialogue-acts.xml"
        _write_xml(path, f"{meeting}.{agent}.dialogue-acts", elements)
    transcript_text = "\n".join(transcripts or []) + "\n"
    (root / "transcripts" / f"{meeting}.trans").write_text(transcript_text, encoding="utf-8")
    return root


def standard_inputs(
    shift: str = "0",
) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str]]:
    """Return words, acts and transcripts covering every C1 situation, with word times optionally shifted."""
    offset = float(shift)

    def at(value: float) -> str:
        return f"{value + offset:g}"

    words_a = [
        word("Mtg001.w.1", "Hello", at(1.0), at(1.2)),
        word("Mtg001.w.2", "there", at(1.2), at(1.4)),
        word("Mtg001.w.3", "yes", at(2.0), at(2.3)),
        word("Mtg001.w.4", "okay", at(2.5), at(2.9)),
        word("Mtg001.w.5", "one", at(4.0), at(4.5)),
        word("Mtg001.w.6", "two", at(5.0), at(5.5)),
        word("Mtg001.w.7", "three", at(6.0), at(7.0)),
        word("Mtg001.w.8", "ok", at(9.0), at(9.4)),
        word("Mtg001.w.9", "um"),
        word("Mtg001.w.1,200", "comma", at(11.0), at(11.4)),
        word("Mtg001.w.10", "café", at(12.0), at(12.4)),
        word("Mtg001.w.11", "so", at(13.0), at(13.1)),
        word("Mtg001.vocalsound.12", "", at(13.1), at(13.3), kind="vocalsound", description="laugh"),
    ]
    words_b = [word("Mtg001.w.101", "right", "1.0", "1.3")]
    acts_a = [
        act("Mtg001.A.dialogueact1", "1.0", "1.4", "s^cs", adjacency="1a", comment="note",
            href=words_href("Mtg001.w.1", "Mtg001.w.2")),
        act("Mtg001.A.dialogueact2", "2.0", "3.0", "s^aa|s^bk", adjacency="1b",
            href=words_href("Mtg001.w.3", "Mtg001.w.4")),
        act("Mtg001.A.dialogueact3", "4.0", "7.0", "s|fg|s^cs",
            href=words_href("Mtg001.w.5", "Mtg001.w.7")),
        act("Mtg001.A.dialogueact4", "8.0", "8.5", "s"),
        act("Mtg001.A.dialogueact5", "9.0", "9.5", None, original_type="s^arp",
            href=words_href("Mtg001.w.8")),
        act("Mtg001.A.dialogueact6", "10.0", "10.0", "s^df", href=words_href("Mtg001.w.9")),
        act("Mtg001.A.dialogueact7", "11.0", "11.4", "s", href=words_href("Mtg001.w.1,200")),
        act("Mtg001.A.dialogueact8", "12.0", "12.5", "s", href=words_href("Mtg001.w.10")),
        act("Mtg001.A.dialogueact9", "13.0", "13.4", "s^aa",
            href=words_href("Mtg001.w.11", "Mtg001.vocalsound.12")),
    ]
    acts_b = [
        act("Mtg001.B.dialogueact1", "1.0", "1.4", "s^co", channel="c2", participant="me002",
            href=words_href("Mtg001.w.101", agent="B")),
    ]
    transcripts = [
        "Mtg001-c1_0002000_0003000,yes | okay",
        "Mtg001-c1_0004000_0007000,one | two | three",
    ]
    return {"A": words_a, "B": words_b}, {"A": acts_a, "B": acts_b}, transcripts


def build_standard_corpus(root: Path, shift: str = "0") -> Path:
    words, acts, transcripts = standard_inputs(shift)
    return write_corpus(root, words=words, acts=acts, transcripts=transcripts)


def build_single_act_corpus(
    root: Path,
    act_xml: str,
    words: list[str] | None = None,
    transcripts: list[str] | None = None,
) -> Path:
    return write_corpus(
        root,
        words={"A": words or []},
        acts={"A": [act_xml]},
        transcripts=transcripts,
    )
