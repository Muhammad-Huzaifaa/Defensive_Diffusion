import torch.nn as nn 
import torch
from utils import get_classifiers_list

class Classifier(nn.Module): 
    """
    MLP classifier. 
    Args:
        num_classes -> number of classes 
        in_feature -> features dimension

    return logits. 
    
    """
    def __init__(self,num_classes=2 ,in_features = 768*196):
        
        super().__init__()
        self.linear1 = nn.Linear(in_features= in_features, out_features= 4096)
        self.linear2 = nn.Linear(in_features= 4096, out_features= 2048)
        self.linear3 = nn.Linear(in_features= 2048, out_features= 128)
        self.linear4 = nn.Linear(in_features= 128, out_features= num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self,x):
        x= x.reshape(-1, 196*768)
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class Big_model(nn.Module): 
 
    def __init__(self, MLP_path = 'models/MLP_new_chest', num_classifiers=3, vit_path='models/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth'):
        super().__init__()
        self.MLP_path = MLP_path
        self.vit_path = vit_path
        self.num_classifiers= num_classifiers
        self.mlp_list = get_classifiers_list(self.MLP_path, num_classifiers = self.num_classifiers)
        self.model = torch.load(self.vit_path)  

    def forward(self,x):
            final_prediction = []
            vit_predictions = self.model(x)
            y = torch.softmax(vit_predictions*25, dim=-1)
            final_prediction.append(y)
            x = self.model.patch_embed(x)
            x_0 = self.model.pos_drop(x)
            i = 0
            for mlp in self.mlp_list:
                x_0 = self.model.blocks[i](x_0)    
                mlp_output = mlp(x_0)
                mlp_predictions = torch.softmax(mlp_output*25, dim=-1)
                final_prediction.append(mlp_predictions)
                i+=1
            stacked_tesnor = torch.stack(final_prediction,dim=1)
            preds_major = stacked_tesnor.sum(dim=1)
            #preds_major = preds_major.float()
            #preds_major = preds_major.requires_grad_(True)
            return preds_major



