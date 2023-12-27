from utils.process import getParams
import random as random
import numpy as np
import torch
from D2CMG import Trainer

dataset = 'acm'  # choice:acm freebase dblp4057 imdb4780
args = getParams(dataset)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)

trainer = Trainer(args)
print(args)
acc, acc_std, macro_f1, macro_f1_std, micro_f1, micro_f1_std, nmi, nmi_std, acc_cluster, acc_cluster_std, st = trainer.training()
print(
    "\t[Classification] Macro-F1: {:.4f} | Micro-F1: {:.4f} | NMI: {:.4f}({:.4f}) | ACC_cluster: {:.4f}({:.4f})".
    format(macro_f1, micro_f1, nmi, nmi_std, acc_cluster, acc_cluster_std))
