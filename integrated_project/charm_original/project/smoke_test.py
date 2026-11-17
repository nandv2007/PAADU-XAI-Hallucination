import numpy as np
from config import Phase1Config
from pipeline_orchestrator import Phase1Pipeline

def main():
    rng=np.random.default_rng(17)
    benign=rng.normal(size=(24,1,10,3)).astype('float32')
    test=rng.normal(size=(4,1,10,3)).astype('float32')
    test[-1,:,:,0]+=7
    cfg=Phase1Config(epochs=1,embed_dim=16,batch_size=8,attribution_top_k=3)
    pipe=Phase1Pipeline(['flow_count','interarrival','payload_stat'],cfg).fit(benign)
    before=pipe.engine.score(test)
    after=pipe.predict(test,metadata=[{'protocol':'unknown'} for _ in test])
    assert [x['anomaly'] for x in before]==[x['anomaly'] for x in after]
    assert all(x['explanation'] is not None for x in after if x['anomaly'])
    print('PASS: original CHARM model + scoring + XAI wiring; decisions unchanged')
    print('alerts:',sum(x['anomaly'] for x in after),'of',len(after))

if __name__=='__main__': main()
