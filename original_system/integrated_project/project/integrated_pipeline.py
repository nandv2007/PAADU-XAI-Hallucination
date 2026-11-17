"""Runnable reconstructed GraphRAG + evidence verification + CHARM boundary.
The student's full source repo is not available. CHARM is unmodified in charm_original/.
"""
import re
from dataclasses import dataclass
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Evidence:
    title: str
    text: str
    score: float

class HybridRetriever:
    def __init__(self, documents):
        self.docs = [(str(d['title']), str(d['text'])) for d in documents]
        if not self.docs: raise ValueError('documents must not be empty')
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
        self.matrix = self.vectorizer.fit_transform([t+' '+v for t,v in self.docs])
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(len(self.docs)))
        for i, (_, text) in enumerate(self.docs):
            for j, (title, _) in enumerate(self.docs):
                if i != j and len(title)>3 and re.search(r'(?<!\w)'+re.escape(title.lower())+r'(?!\w)', text.lower()):
                    self.graph.add_edge(i,j)
    def retrieve(self, query, k=3, hybrid=True):
        scores=cosine_similarity(self.vectorizer.transform([query]),self.matrix).ravel()
        if hybrid:
            boost=np.zeros(len(self.docs))
            for anchor in np.argsort(-scores)[:2]:
                for neighbor in self.graph.neighbors(int(anchor)):
                    boost[neighbor]=max(boost[neighbor],float(scores[anchor])*.22)
            scores=scores+boost
        return [Evidence(self.docs[i][0],self.docs[i][1],float(scores[i])) for i in np.argsort(-scores)[:k] if scores[i]>0]

class EvidenceVerifier:
    """NLI model is optional; no fake entailment probability is returned without it."""
    def __init__(self, model_name='cross-encoder/nli-MiniLM2-L6-H768', load_model=False):
        self.model_name=model_name
        self.tokenizer=self.model=None
        if load_model:
            import torch
            from transformers import AutoTokenizer,AutoModelForSequenceClassification
            self.tokenizer=AutoTokenizer.from_pretrained(model_name)
            self.model=AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    def verify(self, claim, evidence):
        if self.model is None:
            return {'label':'NOT_EVALUATED','reason':'NLI model not loaded','evidence':None}
        import torch
        if not evidence: return {'label':'INSUFFICIENT_EVIDENCE','evidence':None}
        terms=set(re.findall(r'[a-z0-9]+',claim.lower()))-{'the','a','an','and','or','is','are','of','to','in','on','was','were','does','do','not'}
        candidates=[]
        for doc in evidence:
            for sentence in re.split(r'(?<=[.!?])\s+',doc.text):
                overlap=len(terms.intersection(re.findall(r'[a-z0-9]+',sentence.lower())))
                if overlap>=max(1,min(2,len(terms))): candidates.append((overlap,doc.title,sentence))
        if not candidates: return {'label':'INSUFFICIENT_EVIDENCE','evidence':None}
        candidates=sorted(candidates,reverse=True)[:8]
        label_map={int(k):str(v).upper() for k,v in self.model.config.id2label.items()}
        results=[]
        for overlap,title,sentence in candidates:
            batch=self.tokenizer(sentence,claim,return_tensors='pt',truncation=True,max_length=384)
            with torch.no_grad(): probs=torch.softmax(self.model(**batch).logits,dim=-1)[0]
            idx=int(probs.argmax()); label=label_map[idx]
            results.append({'label':label,'confidence':float(probs[idx]),'title':title,'sentence':sentence,'overlap':overlap})
        supported=[x for x in results if 'ENTAIL' in x['label'] and x['confidence']>=.8]
        contradicted=[x for x in results if 'CONTRAD' in x['label'] and x['confidence']>=.9]
        if supported:
            best=max(supported,key=lambda x:x['confidence']); status='SUPPORTED'
        elif contradicted:
            best=max(contradicted,key=lambda x:x['confidence']); status='CONTRADICTED'
        else:
            best=max(results,key=lambda x:x['confidence']); status='INSUFFICIENT_EVIDENCE'
        return {'label':status,'evidence':best}

class IntegratedSystem:
    def __init__(self, documents, enable_nli=False):
        self.retriever=HybridRetriever(documents)
        self.verifier=EvidenceVerifier(load_model=enable_nli)
    def answer(self, query, candidate_answer=None):
        evidence=self.retriever.retrieve(query)
        # Candidate answer is supplied by a caller/LLM. Do not fabricate generated answers.
        result={'query':query,'retrieval':[vars(e) for e in evidence], 'answer':candidate_answer,
                'generation_status':'EXTERNAL_ANSWER_REQUIRED' if candidate_answer is None else 'PROVIDED'}
        if candidate_answer is not None:
            result['verification']=self.verifier.verify(candidate_answer,evidence)
            # Conservative: do not rewrite an answer without independently supported replacement.
            result['corrected_answer']=candidate_answer
            result['correction_status']='UNCHANGED_NO_VERIFIED_REPLACEMENT'
        return result
