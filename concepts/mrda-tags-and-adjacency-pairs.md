# MRDA tags and adjacency pairs

Background only; the rules the project applies are in
[schema 02](../tasks/A-data-transformation/schema/02_annotations_links.md). The sources are the
labelling manual (Dhillon et al., 2004, ICSI TR-04-002; page numbers below refer to it) and the
corpus paper (Shriberg et al., 2004).

## What a label looks like

Each dialogue act in the ICSI MRDA corpus carries a label string built from short tags:

| Example | Reading |
|---|---|
| `s^cs` | A statement (`s`) with an additional tag `cs` attached by `^` |
| `fg\|s` | Two acts in one utterance, separated by `\|`: a floor-grabber, then a statement |
| `s^aa^bk` | One act with two target tags; tags are attached in alphabetical order (p. 15) |
| `s.%--` | A statement with a disruption form after a period: abandoned speech (p. 111) |
| `s^cs:s^co` | A colon marks quoted material; the left side describes the whole utterance, the right side only the quotation (p. 21) |

So four symbols matter: `|` splits an utterance into acts (Task 1 handles it), `^` adds tags, `.`
adds a disruption form, and `:` introduces quotation.

## The classes the project uses

The specification keeps seven classes. Partial accept and partial reject are merged into accept and
reject (pp. 6–7, 59, 62).

| Class | Role |
|---|---|
| `cs`, `co` | The two proposal classes: a suggestion and a command. These are what the project links to replies |
| `cc` | A commitment: an explicit promise to do something (p. 74) |
| `aa`, `ar` | Responses: accept and reject (`aap` and `arp`, the partial forms, are folded into them) |
| `bk` | Acknowledgement or backchannel. The bare tag `b` is the plain backchannel |
| `other` | Everything else |

An act can carry several target tags, for example `s^arp^cs`. The project keeps all of them and
picks one class by a fixed order (see schema 02), because the manual does not rank the classes.

## Tags that are not classes

- `%-` is an interrupted act and `%--` an abandoned one (p. 111). `%` alone describes indecipherable
  speech (p. 110), which is about the audio and not an interruption, so it is an ordinary tag.
- `x` is non-vocal noise (p. 113). When it is the whole label the act is non-speech. As a suffix
  (`s.x`) the act keeps its dialogue act and gains a disruption form (pp. 19–20).
- `z` marks acts that annotators deliberately left unlabelled: chatter before and after the meeting,
  bleeped passages, digit reading and duplicate microphone channels (pp. 17, 113–114). Many of them
  contain transcribed words, and the manual says their adjacency tokens were kept on purpose.

## Adjacency pairs

Some acts are linked as a pair: a first part, usually a proposal or question, and a second part,
the reply. The `adjacency` field encodes this with tokens separated by periods.

| Token | Meaning |
|---|---|
| `5a`, `5b` | The first (`a`) and second (`b`) part of pair number 5 |
| `5b+` | A continuation of the reply (p. 27) |
| `5b-1`, `5b-2` | Replies from different speakers, numbered when several people answer (p. 28) |
| `8_b` | An underscore marks a pair the annotators overlooked (pp. 28–29) |
| `5b.6a` | One act can reply to pair 5 and open pair 6 |

The two parts of a pair must come from different speakers (p. 25), and backchannels are not given
pairs (p. 30). The manual says numbers ascend and are not reused (p. 26), but the field in this
release returns to 1 repeatedly within a meeting. The corpus paper suggests meetings were annotated
in ten-minute chunks, which would explain it. The project therefore closes all open pairs when the
numbering returns to 1.

The manual gives one token to a split act: it attaches to the whole act, not to one of its parts
(p. 27). Several tokens can also appear on a single act, so a token count cannot be matched to a
part count.

## How the project uses this

Task 2 reads the label string and the adjacency string once and stores the result as fields (the
exact rules are in schema 02). The class decides what a model predicts. A proposal with its replying
acts is the target of the linking task: a model must choose which later act, if any, answers it.
