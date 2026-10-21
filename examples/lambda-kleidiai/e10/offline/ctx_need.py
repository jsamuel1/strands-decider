"""Tokens of fork capacity each request needs: one forward holds state + every question of a fork group."""
import json,sys
from tokenizers import Tokenizer
from strands_decider.prompting import render_question, render_state
from strands_decider.schema import SystemOneRequest
from strands_decider.hidden_engine import HiddenStateEngine, HiddenEngineConfig
S=sys.argv[1]
tok=Tokenizer.from_file(S+'/tokenizer.json')
eng=HiddenStateEngine(tok, None, HiddenEngineConfig(max_length=4096), None)
def need(body):
    req=SystemOneRequest.model_validate(body)
    names=list(req.questions); rq=[render_question(req.questions[n]) for n in names]
    s,q,_=eng.fit(render_state(req.state),[r.text for r in rq])
    ql=[len(x) for x in q]
    return len(s), ql
if __name__=='__main__':
    reqs=json.load(open(sys.argv[2]))['requests']
    for v in reqs:
        s,ql=need(v['request'])
        print(v['id'],'state',s,'q',ql,'one-fork',s+sum(ql),'min(state+longest q)',s+max(ql))
