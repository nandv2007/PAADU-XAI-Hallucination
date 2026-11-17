from config import Phase1Config
from charm_engine import CHARMEngine
from xai_interpreter import XAIInterpreter

class Phase1Pipeline:
    def __init__(self, feature_names, config=None):
        self.engine=CHARMEngine(config or Phase1Config(),feature_names)
        self.xai=XAIInterpreter(self.engine)

    def fit(self, benign_blocks):
        self.engine.fit(benign_blocks)
        return self

    def predict(self, blocks, metadata=None, n_rows=None):
        scores=self.engine.score(blocks,n_rows=n_rows)
        return self.xai.explain(blocks,scores,metadata=metadata)
