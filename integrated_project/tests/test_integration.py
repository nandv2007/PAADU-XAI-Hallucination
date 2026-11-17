import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'project'))
from integrated_pipeline import HybridRetriever,IntegratedSystem

def test_retrieval():
    docs=[{'title':'UDP','text':'UDP is connectionless and not reliable.'},{'title':'TCP','text':'TCP provides reliable ordered delivery.'}]
    assert HybridRetriever(docs).retrieve('UDP connectionless')[0].title=='UDP'
    result=IntegratedSystem(docs).answer('UDP connectionless','UDP is connectionless')
    assert result['verification']['label']=='NOT_EVALUATED'
    assert result['corrected_answer']=='UDP is connectionless'
if __name__=='__main__':
    test_retrieval();print('PASS: integration tests')
