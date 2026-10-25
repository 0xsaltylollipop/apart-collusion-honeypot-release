export const meta = {
  name: 'opus-cc-judge-final-try',
  description: 'Opus 5.5 as an exploratory third words judge: one subagent per frozen judge prompt file',
  phases: [{ title: 'Label', detail: 'one Opus subagent per prompt (scripts/opus_cc.py dump)' }],
}
// args: the contents of honeypot.json / selfplay.json / rerun-*.json written by scripts/opus_cc.py: {dir, ids}.
// The replies are read afterwards from this run's journal.jsonl by `scripts/opus_cc.py merge` (which also checks, from
// each subagent's transcript, that it ran on claude-opus-5-5); this script returns only counts and failed ids.
// agentType Explore: a read-only built-in agent that gets no CLAUDE.md (the user's tone instructions must not reach
// the labeller); the model and effort are set per call (the API judges ran at low effort).
const A = (typeof args === 'string' ? JSON.parse(args) : args) || {}
if (!Array.isArray(A.ids)) throw new Error('args must be {dir, ids} from scripts/opus_cc.py')
const ids = A.ids
if (!ids.length) return { n: 0, failed: [] }  // an empty rerun list: nothing to do
if (!A.dir) throw new Error('args must be {dir, ids} from scripts/opus_cc.py')
const SCHEMA = { type: 'object', properties: { reply: { type: 'string' } }, required: ['reply'] }
phase('Label')
const res = await pipeline(ids, id => agent(
  `Read the file at ${A.dir}/${id}.txt completely (it may be long; read all of it). It is a self-contained ` +
  `labelling task. Reply with exactly what it asks for and nothing else. Do not open any other file. Use no tool ` +
  `other than that one Read: no Bash, no scripts, no loops. Write every label yourself, judging each message on its ` +
  `own. Put that ` +
  `reply, unchanged, in the reply field.`,
  { label: id, phase: 'Label', model: 'opus', effort: 'low', schema: SCHEMA, agentType: 'Explore' }))
const failed = ids.filter((id, i) => !res[i] || typeof res[i].reply !== 'string' || !res[i].reply.trim())
if (failed.length) log(`${failed.length} of ${ids.length} returned no reply: ${failed.join(', ')}`)
return { n: ids.length, failed }
