# An empty agent event can hide the current request

By Jabrail Khalil · 6 November 2026

I fixed this in [Google ADK Go #1895](https://github.com/google/adk-go/pull/1895), merged on 5 November in [`6caa3b85`](https://github.com/google/adk-go/commit/6caa3b853ea4d179124f089daf6f39498dc93425). This lab runs a real ADK workflow against that commit and its immediate parent. Both model implementations are local recorders, so the example needs no API key or model service.

The containing stable Go module release has not been verified. The source revisions below make the example independent of that release schedule.

## Two agents, one request

A user asks, “review the previous answer.” A sequential workflow runs `first`, then `reviewer`. The reviewer uses `IncludeContents: "none"`, which selects its current turn rather than the whole conversation.

Suppose the first agent emits a part containing only a thought signature. That part has metadata but no supported payload to relay. The reviewer should still receive the user's request.

On the parent revision, the metadata-only event becomes the current-turn boundary. The earlier user message falls outside the selected history. Filtering the event later cannot recover the message already excluded by that boundary. Empty parts and standalone part metadata have the same problem.

I changed foreign-event conversion to distinguish supported payloads from standalone metadata, then used that result when selecting the boundary. An event with no relay payload is dropped from the model-bound context. The stored event remains intact.

This extends the existing foreign-thought handling from [#1719](https://github.com/google/adk-go/pull/1719). It does not introduce that earlier reasoning filter, remove the current agent's own history, or discard signatures attached to a supported payload.

## Six checks with different jobs

The [Go example](../examples/adk-context/main.go) adapts my earlier [public Runner validation probe](https://github.com/jabrailkhalil/adk-go/blob/6dd62f62abb056a8d83f4c2965bbd65e1b3f8949/.github/validation/runner-probe.go). It imports public ADK packages and exercises `Runner`, `InMemoryService`, `sequentialagent` and `llmagent` together.

| Scenario | Parent revision | Accepted revision |
| --- | --- | --- |
| Empty foreign part | Current user is lost | Current user is retained |
| Foreign signature alone | Current user is lost | Current user is retained |
| Foreign part metadata alone | Current user is lost | Current user is retained |
| Inline payload with a signature, beside empty and metadata-only parts | Extra standalone parts reach the reviewer | Payload and attached signature are retained; standalone parts are omitted |
| Ordinary foreign text | Retained | Retained |
| Reviewer's own thought and history | Retained | Retained |

The first three cases select only the current turn. The payload and history controls use the default history mode. The inline bytes are synthetic text, not an image-quality or provider-media test.

I compare each complete reviewer request with an independently specified expected request. The lab also requires exactly one call to each local model, checks the emitted and stored event counts, and compares snapshots to ensure that context filtering did not mutate stored foreign or own-agent events. It prints structural observations rather than request contents.

## Run it

Install Git, Go 1.26.10 and Python 3.12. From this repository's root, run the same command in PowerShell or a Unix shell:

```sh
python examples/adk-context/run.py
```

On a system with a working Go race-detector toolchain:

```sh
python examples/adk-context/run.py --race
```

The runner fetches these immutable revisions into `.lab/adk-go`:

- Parent: `a1603c7f179af285b08ef4f03e757a6deb0630b9`
- Accepted change: `6caa3b853ea4d179124f089daf6f39498dc93425`

It uses each revision's own `go.mod` and `go.sum` with `-mod=readonly`. Git and Go need network access to obtain source and dependencies; the workflow itself makes no model calls. A dirty existing lab checkout stops the runner.

The JSON files appear under `.lab/adk-baseline.json` and `.lab/adk-fixed.json`. A successful baseline reproduction reports six expected behaviors matched but only two fixed contracts matched. The accepted revision must match all six fixed contracts.

The runner also deliberately supplies the opposite expectation to each binary. Both commands must fail, and the [result checker](../examples/adk-context/check_results.py) requires exactly the four named request-contract differences. A timeout, panic, storage mutation or unexpected model call cannot count as successful bug reproduction. Invalid command modes must fail before the scenarios run.

The [recorded Linux and Windows measurements](../validation/adk-context/measurements.json) contain both successful revision checks and both deliberately wrong expectations. Linux used the race detector. The source revision and CI run are recorded alongside the JSON.

The [CI workflow](../.github/workflows/adk-context.yml) runs the same entry point on Linux with the race detector and on Windows without it. It checks formatting and `go vet` and uploads the measured JSON. This validates these source-level model-bound requests. It does not establish acceptance by a live provider, a cloud deployment, database-backed sessions or measured production impact.

## A short exercise

Before running the lab, predict the reviewer's request in the signature-only case. Then replace that signature with a text payload and explain why a visible foreign answer is a different boundary.

Finally, remove the signature from the inline-payload fixture. The payload should still be relayed. Put the signature back and check that both the payload and its attached metadata survive. This distinguishes metadata alone having no relay payload from removing all metadata.

I prepared this as self-study material and a possible lesson exercise. No delivered class, audience or downstream adoption is claimed.
