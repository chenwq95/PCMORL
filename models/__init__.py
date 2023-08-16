from models.updown import UpDown
from models.xlan import XLAN
from models.pe_xlan import PEXLAN
from models.xtransformer import XTransformer
from models.transLstm import TransLSTM
from models.xlan_MTL import XLANMTL
from models.variational_xlan import VariationalXLAN
from models.vanilla_cvae_xlan import VanillaCVAEXLAN
from models.cvae_xlan import CVAEXLAN
from models.cvae_xlan_diverse import CVAEXLANDiverse
from models.PACVAE import PACVAE

#common baselines
from models.LSTMAtt import LSTMAtt
from models.Transformer import Transformer
from models.TransformerV2 import TransformerV2


from models.vanilla_cvae_xlan_diverse import VanillaCVAEXLANDiverse

from models.gmcvae_xlan_diverse import GMCVAEXLANDiverse
from models.variational_xlan_diverse import VariationalXLANDiverse

__factory = {
    'UpDown': UpDown,
    'XLAN': XLAN,
    'PEXLAN': PEXLAN,
    'XTransformer': XTransformer,
    'TransLSTM': TransLSTM,
    'XLANMTL': XLANMTL,
    'VariationalXLAN': VariationalXLAN,
    'VanillaCVAEXLAN': VanillaCVAEXLAN,
    'CVAEXLAN': CVAEXLAN,
    'CVAEXLANDiverse': CVAEXLANDiverse,
    'LSTMAtt': LSTMAtt,
    'Transformer': Transformer,
    'TransformerV2' : TransformerV2,
    'VanillaCVAEXLANDiverse' : VanillaCVAEXLANDiverse,
    'GMCVAEXLANDiverse' : GMCVAEXLANDiverse,
    'VariationalXLANDiverse' : VariationalXLANDiverse,
    'PACVAE' : PACVAE
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](*args, **kwargs)