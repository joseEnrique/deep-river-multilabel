from typing import List, Dict, Type, Union
import statistics
from river import base
from classes.rolling_multilabel_classifier import RollingMultiLabelClassifier
import torch

class RollingEnsembleClassifier(base.MultiLabelClassifier):
    """
    Ensemble of RollingMultiLabelClassifiers.
    Trains 'n_models' independently (with different seeds) and averages their predictions.
    This reduces variance and usually improves performance in online settings.
    """
    def __init__(
        self,
        n_models: int = 3,
        module: Type[torch.nn.Module] = None,
        label_names: List[str] = None,
        device: Union[str, List[str]] = "cpu",
        member_cls: Type[base.MultiLabelClassifier] = RollingMultiLabelClassifier,
        **kwargs
    ):
        self.models = []
        self.label_names = label_names
        # kwargs usually contains 'seed'. We want to vary it.
        base_seed = kwargs.pop('seed', 42)
        
        # Determine device for each model
        if isinstance(device, list):
            if len(device) != n_models:
                # If list length doesn't match n_models, cycle through them
                devices = [device[i % len(device)] for i in range(n_models)]
            else:
                devices = device
        else:
            devices = [device] * n_models

        for i in range(n_models):
            # Create a unique seed for each member
            member_seed = base_seed + i
            clf = member_cls(
                module=module,
                label_names=label_names,
                device=devices[i],
                seed=member_seed,
                **kwargs
            )
            self.models.append(clf)

    def learn_one(self, x: Dict, y: Dict, **kwargs):
        # Train all models on the same instance
        for model in self.models:
            model.learn_one(x, y, **kwargs)
        return self

    def predict_proba_one(self, x: Dict) -> Dict[str, float]:
        # Collect predictions from all models
        all_probs = [model.predict_proba_one(x) for model in self.models]
        
        # Average probabilities
        avg_probs = {}
        for label in self.label_names:
            probs_for_label = [p[label] for p in all_probs]
            avg_probs[label] = statistics.mean(probs_for_label)
            
        return avg_probs

    def predict_one(self, x: Dict) -> Dict[str, int]:
        # Use simple 0.5 threshold on the averaged probabilities
        # (Or could use individual thresholds, but averaging is cleaner)
        probas = self.predict_proba_one(x)
        # Note: Ideally we use the thresholds from the first model or a global one.
        # Let's assume passed thresholds in kwargs were applied to members.
        # But predict_one logic in RollingMultiLabelClassifier uses its internal threshold.
        # Here we just use > 0.5 or we can expose a threshold param.
        # The user plan mentions "Optimize Thresholds".
        # We'll stick to > 0.5 here unless we support custom logic.
        # Actually, let's grab the threshold from model[0] defined in kwargs
        thresh_dict = self.models[0].thresholds
        return {l: int(p >= thresh_dict.get(l, 0.5)) for l, p in probas.items()}
