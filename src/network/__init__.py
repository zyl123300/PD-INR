from .network import DensityNetwork, GCNNetwork, OneLayerMLP


# , DensityNetworkTCNN


def get_network(type):
    if type == "mlp":
        return DensityNetwork
    elif type == "gcn":
        return GCNNetwork
    elif type == "1layermlp":
        return OneLayerMLP
    else:
        raise NotImplementedError("Unknown network typeß!")

