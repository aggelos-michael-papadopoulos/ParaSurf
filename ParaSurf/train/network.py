import numpy as np
import os
import torch
from ParaSurf.train.features import KalasantyFeaturizer
from ParaSurf.model import ParaSurf_model



class Network:
    def __init__(self, model_path, gridSize, feature_channels, voxelSize=1, device="cuda"):
        self.gridSize = gridSize                                   # Does this change?

        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # load model
        self.model = ParaSurf_model.ResNet3D_Transformer(in_channels=feature_channels, block=ParaSurf_model.DilatedBottleneck,
                                                          num_blocks=[3, 4, 6, 3], num_classes=1)

        # load weights
        self.model.load_state_dict(torch.load(os.path.join(model_path)))
        # model to eval mode and to device
        self.model = self.model.to(self.device).eval()

        self.featurizer = KalasantyFeaturizer(gridSize, voxelSize) # it is the "rules of the game"
        self.feature_channels = feature_channels
        
    def get_lig_scores(self, prot, batch_size, add_forcefields, add_atom_radius_features):

        self.featurizer.get_channels(prot.mol, add_forcefields, add_atom_radius_features)


        lig_scores = []
        input_data = torch.zeros((batch_size, self.gridSize, self.gridSize, self.gridSize, self.feature_channels), device=self.device)

        batch_cnt = 0
        for p, n in zip(prot.surf_points, prot.surf_normals):
            input_data[batch_cnt,:,:,:,:] = torch.tensor(self.featurizer.grid_feats(p, n, prot.heavy_atom_coords), device=self.device)
            batch_cnt += 1
            if batch_cnt == batch_size:
                with torch.no_grad():
                    output = self.model(input_data)
                    output = torch.sigmoid(output)
                lig_scores.extend(output.cpu().numpy())
                batch_cnt = 0

        if batch_cnt > 0:
            with torch.no_grad():
                output = self.model(input_data[:batch_cnt])
                output = torch.sigmoid(output)
            lig_scores.extend(output.cpu().numpy())

        print(np.array(lig_scores).shape)
        return np.array(lig_scores)


